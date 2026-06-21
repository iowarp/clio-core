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
 * FUSE ADAPTER UNIT TESTS
 *
 * Tests the FUSE adapter's CTE-backed filesystem operations:
 * 1. Tag lifecycle (create, exists, delete)
 * 2. Implicit directory discovery via TagQuery
 * 3. Page-based I/O round-trip (single page, multi-page, cross-page)
 * 4. End-to-end integration (create file, write, read, list, delete)
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "adapter/libfuse/fuse_cte.h"
#include "simple_test.h"

namespace fs = std::filesystem;
using namespace clio::cae::fuse;

// ============================================================================
// Test fixture
// ============================================================================

class FuseAdapterTestFixture {
 public:
  static constexpr clio::run::u64 kTestTargetSize = 1024 * 1024 * 10;  // 10MB

  std::string test_storage_path_;
  static inline bool g_initialized = false;
  bool target_initialized_ = false;

  FuseAdapterTestFixture() {
    std::string home_dir = ctp::SystemInfo::GetHomeDir();
    REQUIRE(!home_dir.empty());
    test_storage_path_ = home_dir + "/cte_fuse_test.dat";

    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
    }

    if (!g_initialized) {
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      REQUIRE(success);

      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
      REQUIRE(success);

      auto *cte_client = CLIO_CTE_CLIENT;
      REQUIRE(cte_client != nullptr);
      cte_client->Init(clio::cte::core::kCtePoolId);

      clio::cte::core::CreateParams params;
      auto create_task = cte_client->AsyncCreate(
          clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
          clio::cte::core::kCtePoolId, params);
      create_task.Wait();
      REQUIRE(create_task->GetReturnCode() == 0);

      g_initialized = true;
      INFO("CTE runtime initialized for FUSE adapter tests");
    }
  }

  ~FuseAdapterTestFixture() {
    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
    }
  }

  void SetupTarget() {
    if (target_initialized_) {
      return;
    }

    auto *cte_client = CLIO_CTE_CLIENT;

    clio::run::PoolId bdev_pool_id(950, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create_task =
        bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(), test_storage_path_,
                                bdev_pool_id, clio::run::bdev::BdevType::kFile);
    create_task.Wait();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto reg_task = cte_client->AsyncRegisterTarget(
        test_storage_path_, clio::run::bdev::BdevType::kFile, kTestTargetSize,
        clio::run::PoolQuery::Local(), bdev_pool_id);
    reg_task.Wait();
    REQUIRE(reg_task->GetReturnCode() == 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    target_initialized_ = true;
    INFO("Storage target registered for FUSE adapter tests");
  }

  std::vector<char> CreateTestData(size_t size, char pattern = 'F') {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(pattern + (i % 26));
    }
    return data;
  }

  bool VerifyTestData(const std::vector<char> &data, char pattern = 'F') {
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] != static_cast<char>(pattern + (i % 26))) {
        return false;
      }
    }
    return true;
  }

  /** Helper to clean up a tag, ignoring errors */
  void CleanupTag(const std::string &name) {
    CteDelTag(name);
  }
};

// ============================================================================
// Tag lifecycle tests
// ============================================================================

TEST_CASE("FUSE CTE - Tag create and exists", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string tag_name = "/fuse_test/tag_exists";

  // Should not exist yet
  REQUIRE_FALSE(CteTagExists(tag_name));

  // Create it
  auto tag_id = CteGetOrCreateTag(tag_name);
  REQUIRE(!tag_id.IsNull());

  // Should exist now
  REQUIRE(CteTagExists(tag_name));

  INFO("Tag create and exists verified");
  fixture->CleanupTag(tag_name);
}

TEST_CASE("FUSE CTE - Tag deletion", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string tag_name = "/fuse_test/tag_delete";

  auto tag_id = CteGetOrCreateTag(tag_name);
  REQUIRE(!tag_id.IsNull());
  REQUIRE(CteTagExists(tag_name));

  CteDelTag(tag_name);

  // Should no longer exist
  REQUIRE_FALSE(CteTagExists(tag_name));
  INFO("Tag deletion verified");
}

// ============================================================================
// Implicit directory tests via TagQuery
// ============================================================================

TEST_CASE("FUSE CTE - CteDirExists for implicit directories", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  // Create tags that imply directory structure
  std::string file1 = "/fuse_dir_test/a/b/file1.txt";
  std::string file2 = "/fuse_dir_test/a/b/file2.txt";
  std::string file3 = "/fuse_dir_test/a/c/file3.txt";

  auto id1 = CteGetOrCreateTag(file1);
  auto id2 = CteGetOrCreateTag(file2);
  auto id3 = CteGetOrCreateTag(file3);
  REQUIRE(!id1.IsNull());
  REQUIRE(!id2.IsNull());
  REQUIRE(!id3.IsNull());

  // These implicit directories should exist
  REQUIRE(CteDirExists("/fuse_dir_test"));
  REQUIRE(CteDirExists("/fuse_dir_test/a"));
  REQUIRE(CteDirExists("/fuse_dir_test/a/b"));
  REQUIRE(CteDirExists("/fuse_dir_test/a/c"));

  // This should not
  REQUIRE_FALSE(CteDirExists("/fuse_dir_test/a/d"));
  REQUIRE_FALSE(CteDirExists("/fuse_dir_test/nonexistent"));

  INFO("CteDirExists verified for implicit directories");
  fixture->CleanupTag(file1);
  fixture->CleanupTag(file2);
  fixture->CleanupTag(file3);
}

TEST_CASE("FUSE CTE - CteListDirectChildren", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string f1 = "/fuse_list_test/alpha.txt";
  std::string f2 = "/fuse_list_test/beta.txt";
  std::string f3 = "/fuse_list_test/sub/gamma.txt";  // NOT a direct child of /fuse_list_test

  auto id1 = CteGetOrCreateTag(f1);
  auto id2 = CteGetOrCreateTag(f2);
  auto id3 = CteGetOrCreateTag(f3);
  REQUIRE(!id1.IsNull());
  REQUIRE(!id2.IsNull());
  REQUIRE(!id3.IsNull());

  auto children = CteListDirectChildren("/fuse_list_test");
  REQUIRE(children.size() == 2);
  REQUIRE(std::find(children.begin(), children.end(), "alpha.txt") != children.end());
  REQUIRE(std::find(children.begin(), children.end(), "beta.txt") != children.end());
  // gamma.txt is under /fuse_list_test/sub/, not a direct child
  REQUIRE(std::find(children.begin(), children.end(), "gamma.txt") == children.end());

  INFO("CteListDirectChildren verified: " << children.size() << " direct children");
  fixture->CleanupTag(f1);
  fixture->CleanupTag(f2);
  fixture->CleanupTag(f3);
}

TEST_CASE("FUSE CTE - CteListSubdirs", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string f1 = "/fuse_subdir_test/x/file1.txt";
  std::string f2 = "/fuse_subdir_test/x/file2.txt";
  std::string f3 = "/fuse_subdir_test/y/file3.txt";
  std::string f4 = "/fuse_subdir_test/direct.txt";  // Not in a subdirectory

  auto id1 = CteGetOrCreateTag(f1);
  auto id2 = CteGetOrCreateTag(f2);
  auto id3 = CteGetOrCreateTag(f3);
  auto id4 = CteGetOrCreateTag(f4);

  auto subdirs = CteListSubdirs("/fuse_subdir_test");
  // Should find "x" and "y" (deduplicated)
  REQUIRE(subdirs.size() == 2);
  REQUIRE(std::find(subdirs.begin(), subdirs.end(), "x") != subdirs.end());
  REQUIRE(std::find(subdirs.begin(), subdirs.end(), "y") != subdirs.end());

  INFO("CteListSubdirs verified: " << subdirs.size() << " subdirectories");
  fixture->CleanupTag(f1);
  fixture->CleanupTag(f2);
  fixture->CleanupTag(f3);
  fixture->CleanupTag(f4);
}

// ============================================================================
// Page-based I/O tests
// ============================================================================

TEST_CASE("FUSE CTE - Small write and read round-trip", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string tag_name = "/fuse_io_test/small_rw";
  auto tag_id = CteGetOrCreateTag(tag_name);
  REQUIRE(!tag_id.IsNull());

  const size_t data_size = 128;
  auto write_data = fixture->CreateTestData(data_size, 'S');

  REQUIRE(CtePutBlob(tag_id, "0", write_data.data(), data_size, 0));

  std::vector<char> read_data(data_size);
  REQUIRE(CteGetBlob(tag_id, "0", read_data.data(), data_size, 0));
  REQUIRE(fixture->VerifyTestData(read_data, 'S'));

  INFO("Small write/read round-trip verified: " << data_size << " bytes");
  fixture->CleanupTag(tag_name);
}

TEST_CASE("FUSE CTE - Multi-page write and read", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string tag_name = "/fuse_io_test/multipage_rw";
  auto tag_id = CteGetOrCreateTag(tag_name);
  REQUIRE(!tag_id.IsNull());

  const size_t total_size = kDefaultPageSize * 3;
  auto write_data = fixture->CreateTestData(total_size, 'M');

  // Write page by page
  for (size_t p = 0; p < 3; ++p) {
    REQUIRE(CtePutBlob(tag_id, std::to_string(p),
                       write_data.data() + p * kDefaultPageSize,
                       kDefaultPageSize, 0));
  }

  // Read back page by page
  std::vector<char> read_data(total_size);
  for (size_t p = 0; p < 3; ++p) {
    REQUIRE(CteGetBlob(tag_id, std::to_string(p),
                       read_data.data() + p * kDefaultPageSize,
                       kDefaultPageSize, 0));
  }

  REQUIRE(fixture->VerifyTestData(read_data, 'M'));

  // Verify tag size
  size_t tag_size = CteGetTagSize(tag_id);
  REQUIRE(tag_size == total_size);

  INFO("Multi-page round-trip verified: " << total_size << " bytes, 3 pages");
  fixture->CleanupTag(tag_name);
}

TEST_CASE("FUSE CTE - Partial page write with offset", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string tag_name = "/fuse_io_test/partial_page";
  auto tag_id = CteGetOrCreateTag(tag_name);
  REQUIRE(!tag_id.IsNull());

  const size_t data_size = 100;
  const size_t blob_offset = 50;
  auto write_data = fixture->CreateTestData(data_size, 'P');

  REQUIRE(CtePutBlob(tag_id, "0", write_data.data(), data_size, blob_offset));

  std::vector<char> read_data(data_size);
  REQUIRE(CteGetBlob(tag_id, "0", read_data.data(), data_size, blob_offset));
  REQUIRE(fixture->VerifyTestData(read_data, 'P'));

  INFO("Partial page write/read with offset verified");
  fixture->CleanupTag(tag_name);
}

TEST_CASE("FUSE CTE - Cross-page write simulation", "[fuse][cte]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  std::string tag_name = "/fuse_io_test/cross_page";
  auto tag_id = CteGetOrCreateTag(tag_name);
  REQUIRE(!tag_id.IsNull());

  // Write 200 bytes starting at offset 4000 (page boundary at 4096)
  // Page 0: 96 bytes at offset 4000, Page 1: 104 bytes at offset 0
  const size_t total_write = 200;
  const size_t file_offset = kDefaultPageSize - 96;
  auto write_data = fixture->CreateTestData(total_write, 'C');

  size_t page0_offset = file_offset % kDefaultPageSize;
  size_t page0_size = kDefaultPageSize - page0_offset;
  size_t page1_size = total_write - page0_size;

  REQUIRE(CtePutBlob(tag_id, "0", write_data.data(), page0_size, page0_offset));
  REQUIRE(CtePutBlob(tag_id, "1", write_data.data() + page0_size, page1_size, 0));

  // Read back
  std::vector<char> read_data(total_write);
  REQUIRE(CteGetBlob(tag_id, "0", read_data.data(), page0_size, page0_offset));
  REQUIRE(CteGetBlob(tag_id, "1", read_data.data() + page0_size, page1_size, 0));

  REQUIRE(fixture->VerifyTestData(read_data, 'C'));
  INFO("Cross-page write/read verified: " << total_write << " bytes spanning pages 0-1");
  fixture->CleanupTag(tag_name);
}

// ============================================================================
// End-to-end integration
// ============================================================================

TEST_CASE("FUSE Integration - Full file lifecycle", "[fuse][integration]") {
  auto *fixture = ctp::Singleton<FuseAdapterTestFixture>::GetInstance();
  fixture->SetupTarget();

  // 1. Create file (tag)
  std::string file_path = "/fuse_e2e/data/test.bin";
  auto tag_id = CteGetOrCreateTag(file_path);
  REQUIRE(!tag_id.IsNull());

  // 2. Verify implicit directories exist
  REQUIRE(CteDirExists("/fuse_e2e"));
  REQUIRE(CteDirExists("/fuse_e2e/data"));

  // 3. Write data spanning 2 pages (5000 bytes)
  const size_t total_size = 5000;
  auto write_data = fixture->CreateTestData(total_size, 'E');

  size_t bytes_written = 0;
  size_t cur = 0;
  while (bytes_written < total_size) {
    size_t page = cur / kDefaultPageSize;
    size_t poff = cur % kDefaultPageSize;
    size_t to_write = std::min(kDefaultPageSize - poff, total_size - bytes_written);
    REQUIRE(CtePutBlob(tag_id, std::to_string(page),
                       write_data.data() + bytes_written, to_write, poff));
    bytes_written += to_write;
    cur += to_write;
  }

  // 4. Verify tag size
  REQUIRE(CteGetTagSize(tag_id) == total_size);

  // 5. Read data back
  std::vector<char> read_data(total_size);
  size_t bytes_read = 0;
  cur = 0;
  while (bytes_read < total_size) {
    size_t page = cur / kDefaultPageSize;
    size_t poff = cur % kDefaultPageSize;
    size_t to_read = std::min(kDefaultPageSize - poff, total_size - bytes_read);
    REQUIRE(CteGetBlob(tag_id, std::to_string(page),
                       read_data.data() + bytes_read, to_read, poff));
    bytes_read += to_read;
    cur += to_read;
  }
  REQUIRE(fixture->VerifyTestData(read_data, 'E'));

  // 6. Verify listing
  auto children = CteListDirectChildren("/fuse_e2e/data");
  REQUIRE(children.size() == 1);
  REQUIRE(children[0] == "test.bin");

  auto subdirs = CteListSubdirs("/fuse_e2e");
  REQUIRE(subdirs.size() == 1);
  REQUIRE(subdirs[0] == "data");

  // 7. Delete and verify gone
  CteDelTag(file_path);
  REQUIRE_FALSE(CteTagExists(file_path));
  REQUIRE_FALSE(CteDirExists("/fuse_e2e"));

  INFO("Full file lifecycle verified: create, write, read, list, delete");
}

SIMPLE_TEST_MAIN()
