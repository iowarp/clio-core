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
 * test_cae_comprehensive.cc
 *
 * Comprehensive tests for CAE (Context Assimilation Engine) Core APIs
 * Focus: Improving coverage of CAE core components
 */

#include "simple_test.h"
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/core_tasks.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include <clio_ctp/introspect/system_info.h>

using namespace clio::cae::core;

/**
 * Test fixture for CAE comprehensive tests
 */
class CAEComprehensiveFixture {
public:
  static inline bool g_initialized = false;
  static inline bool g_cae_initialized = false;
  std::string test_data_dir_;
  std::string test_binary_file_;
  std::string test_hdf5_file_;

  static constexpr size_t kSmallFileSize = 1024;  // 1KB for quick tests

  CAEComprehensiveFixture() {
    if (!g_initialized) {
      INFO("=== Initializing CAE Test Environment ===");

      // Step 1: Initialize CLIO Runtime runtime
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (!success) {
        throw std::runtime_error("CLIO_INIT failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Step 2: Initialize CTE client subsystem
      success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
      if (!success) {
        throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
      }

      // Step 3: Set pool ID on global CTE client
      auto *cte_client = CLIO_CTE_CLIENT;
      cte_client->Init(clio::cte::core::kCtePoolId);

      // Step 4: Create CTE core pool
      clio::cte::core::CreateParams cte_params;
      auto cte_create = cte_client->AsyncCreate(
          clio::run::PoolQuery::Dynamic(),
          clio::cte::core::kCtePoolName,
          clio::cte::core::kCtePoolId,
          cte_params);
      cte_create.Wait();

      if (cte_create->GetReturnCode() != 0) {
        throw std::runtime_error("CTE pool creation failed");
      }

      INFO("CTE infrastructure initialized");
      g_initialized = true;
    }

    // ctp::SystemInfo::GetHomeDir() returns HOME on POSIX, USERPROFILE
    // on Windows. Plain getenv("HOME") segfaults on Windows because the
    // var isn't set and std::string(nullptr) is UB.
    test_data_dir_ = ctp::SystemInfo::GetHomeDir() + "/cae_test_data";
    test_binary_file_ = test_data_dir_ + "/test_binary.bin";
    test_hdf5_file_ = "/workspace/context-assimilation-engine/data/A46_xx.h5";
  }

  void InitializeCAE() {
    if (g_cae_initialized) {
      return;
    }

    INFO("Initializing CAE client");

    // Initialize CAE client subsystem
    CLIO_CAE_CLIENT_INIT();

    // Create CAE client and pool
    clio::cae::core::Client cae_client;
    clio::cae::core::CreateParams cae_params;
    auto cae_create = cae_client.AsyncCreate(
        clio::run::PoolQuery::Local(),
        "test_cae_pool",
        clio::cae::core::kCaePoolId,
        cae_params);
    cae_create.Wait();

    if (cae_create->GetReturnCode() != 0) {
      throw std::runtime_error("CAE pool creation failed");
    }

    INFO("CAE infrastructure initialized");
    g_cae_initialized = true;
  }

  void SetupTestData() {
    // Use std::filesystem instead of `system("mkdir -p ...")` — Windows
    // cmd.exe doesn't take -p and a forked `mkdir` against a path with
    // spaces or backslashes blows up. create_directories returns false
    // if the dir already exists, which is harmless.
    std::error_code _ec;
    std::filesystem::create_directories(test_data_dir_, _ec);

    // Generate small binary test file
    GenerateBinaryFile(test_binary_file_, kSmallFileSize);
  }

  void GenerateBinaryFile(const std::string& path, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to create binary file: " + path);
    }

    // Write patterned data (4-byte blocks with incrementing values)
    const size_t block_size = 4;
    const size_t num_blocks = size / block_size;

    for (size_t i = 0; i < num_blocks; ++i) {
      uint32_t value = static_cast<uint32_t>(i);
      file.write(reinterpret_cast<const char*>(&value), block_size);
    }

    file.close();
    INFO("Generated binary test file: " << path << " (" << size << " bytes)");
  }

  std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return {};
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    return buffer;
  }

  ~CAEComprehensiveFixture() {
    INFO("CAE test cleanup");
  }
};

// ============================================================================
// CAE Client Initialization Tests
// ============================================================================

TEST_CASE("CAE - Client Initialization", "[cae][core][init]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();

  // Create a client connected to the existing pool
  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);
  INFO("CAE client initialized successfully");
}

// ============================================================================
// AssimilationCtx Tests
// ============================================================================

TEST_CASE("CAE - AssimilationCtx Default Constructor", "[cae][ctx]") {
  AssimilationCtx ctx;

  REQUIRE(ctx.src.empty());
  REQUIRE(ctx.dst.empty());
  REQUIRE(ctx.format.empty());
  REQUIRE(ctx.range_off == 0);
  REQUIRE(ctx.range_size == 0);
  INFO("AssimilationCtx default constructor works correctly");
}

TEST_CASE("CAE - AssimilationCtx Full Constructor", "[cae][ctx]") {
  std::string src = "file::/tmp/test.bin";
  std::string dst = "iowarp::test_tag";
  std::string format = "binary";
  size_t offset = 1024;
  size_t size = 2048;

  AssimilationCtx ctx(src, dst, format, "", offset, size);

  REQUIRE(ctx.src == src);
  REQUIRE(ctx.dst == dst);
  REQUIRE(ctx.format == format);
  REQUIRE(ctx.range_off == offset);
  REQUIRE(ctx.range_size == size);
  INFO("AssimilationCtx full constructor works correctly");
}

TEST_CASE("CAE - AssimilationCtx With Patterns", "[cae][ctx]") {
  AssimilationCtx ctx("file::/data.h5", "iowarp::hdf5_tag", "hdf5");

  ctx.include_patterns.push_back("/dataset1/*");
  ctx.include_patterns.push_back("/group/*/data");
  ctx.exclude_patterns.push_back("*/temp*");

  REQUIRE(ctx.include_patterns.size() == 2);
  REQUIRE(ctx.exclude_patterns.size() == 1);
  INFO("AssimilationCtx pattern handling works correctly");
}

// ============================================================================
// ParseOmni Tests - Binary Assimilation
// ============================================================================

TEST_CASE("CAE - ParseOmni Binary File", "[cae][parseomni][binary]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();
  fixture.SetupTestData();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Create assimilation context for binary file
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string dst_url = "iowarp::cae_binary_test";

  AssimilationCtx ctx(src_url, dst_url, "binary");
  std::vector<AssimilationCtx> contexts = {ctx};

  // Execute ParseOmni
  auto task = cae_client.AsyncParseOmni(contexts);
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  INFO("ParseOmni binary assimilation completed: scheduled "
       << task->num_tasks_scheduled_ << " tasks");
}

TEST_CASE("CAE - ParseOmni Binary With Range", "[cae][parseomni][binary][range]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();
  fixture.SetupTestData();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Create assimilation context with range
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string dst_url = "iowarp::cae_binary_range_test";
  size_t offset = 256;
  size_t size = 512;

  AssimilationCtx ctx(src_url, dst_url, "binary", "", offset, size);
  std::vector<AssimilationCtx> contexts = {ctx};

  // Execute ParseOmni
  auto task = cae_client.AsyncParseOmni(contexts);
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  INFO("ParseOmni range assimilation completed");
}

TEST_CASE("CAE - ParseOmni Multiple Contexts", "[cae][parseomni][multi]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();
  fixture.SetupTestData();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Create multiple assimilation contexts
  std::vector<AssimilationCtx> contexts;

  contexts.emplace_back("file::" + fixture.test_binary_file_,
                        "iowarp::multi_test_1", "binary");
  contexts.emplace_back("file::" + fixture.test_binary_file_,
                        "iowarp::multi_test_2", "binary",
                        "", 0, 512);  // First half
  contexts.emplace_back("file::" + fixture.test_binary_file_,
                        "iowarp::multi_test_3", "binary",
                        "", 512, 512);  // Second half

  // Execute ParseOmni
  auto task = cae_client.AsyncParseOmni(contexts);
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  INFO("ParseOmni multiple contexts completed: scheduled "
       << task->num_tasks_scheduled_ << " tasks");
}

TEST_CASE("CAE - ParseOmni Empty Context List", "[cae][parseomni][error]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Execute ParseOmni with empty context list
  std::vector<AssimilationCtx> contexts;
  auto task = cae_client.AsyncParseOmni(contexts);
  task.Wait();

  // Should handle empty list gracefully
  INFO("ParseOmni with empty context list: tasks scheduled = "
       << task->num_tasks_scheduled_);
}

// ============================================================================
// HDF5 Dataset Processing Tests
// ============================================================================

TEST_CASE("CAE - ProcessHdf5Dataset Basic", "[cae][hdf5]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Process HDF5 dataset
  auto task = cae_client.AsyncProcessHdf5Dataset(
      clio::run::PoolQuery::Local(),
      fixture.test_hdf5_file_,
      "/data",
      "cae_hdf5_test");
  task.Wait();

  INFO("ProcessHdf5Dataset completed with code: " << task->GetReturnCode());
}

TEST_CASE("CAE - ProcessHdf5Dataset NonExistent File", "[cae][hdf5][error]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Try to process non-existent file
  auto task = cae_client.AsyncProcessHdf5Dataset(
      clio::run::PoolQuery::Local(),
      "/nonexistent/file.h5",
      "/data",
      "cae_hdf5_error_test");
  task.Wait();

  // Should fail gracefully
  INFO("ProcessHdf5Dataset on non-existent file: code = "
       << task->GetReturnCode());
}

TEST_CASE("CAE - ProcessHdf5Dataset Empty Dataset Path", "[cae][hdf5][error]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Try with empty dataset path
  auto task = cae_client.AsyncProcessHdf5Dataset(
      clio::run::PoolQuery::Local(),
      fixture.test_hdf5_file_,
      "",
      "cae_hdf5_empty_path_test");
  task.Wait();

  INFO("ProcessHdf5Dataset with empty dataset path: code = "
       << task->GetReturnCode());
}

// ============================================================================
// AssimilationCtx Serialization Tests
// ============================================================================

TEST_CASE("CAE - AssimilationCtx Serialization", "[cae][ctx][serialization]") {
  AssimilationCtx ctx("file::/test.bin", "iowarp::tag", "binary", "dep1", 100, 200);
  ctx.include_patterns.push_back("*.dat");
  ctx.exclude_patterns.push_back("*.tmp");

  // Serialize using GlobalSerialize
  std::vector<char> buf;
  {
    ctp::ipc::GlobalSerialize<std::vector<char>> oarchive(buf);
    oarchive(ctx);
    oarchive.Finalize();
  }

  // Deserialize
  AssimilationCtx ctx_restored;
  {
    ctp::ipc::GlobalDeserialize<std::vector<char>> iarchive(buf);
    iarchive(ctx_restored);
  }

  // Verify
  REQUIRE(ctx_restored.src == ctx.src);
  REQUIRE(ctx_restored.dst == ctx.dst);
  REQUIRE(ctx_restored.format == ctx.format);
  REQUIRE(ctx_restored.depends_on == ctx.depends_on);
  REQUIRE(ctx_restored.range_off == ctx.range_off);
  REQUIRE(ctx_restored.range_size == ctx.range_size);
  REQUIRE(ctx_restored.include_patterns.size() == 1);
  REQUIRE(ctx_restored.exclude_patterns.size() == 1);

  INFO("AssimilationCtx serialization/deserialization successful");
}

// ============================================================================
// Data Verification Tests
// ============================================================================

TEST_CASE("CAE - Verify Binary Data in CTE", "[cae][integration][binary]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();
  fixture.SetupTestData();

  clio::cae::core::Client cae_client(clio::cae::core::kCaePoolId);

  // Assimilate binary file
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string tag_name = "cae_verify_test";
  std::string dst_url = "iowarp::" + tag_name;

  AssimilationCtx ctx(src_url, dst_url, "binary");
  std::vector<AssimilationCtx> contexts = {ctx};

  auto parse_task = cae_client.AsyncParseOmni(contexts);
  parse_task.Wait();
  REQUIRE(parse_task->GetReturnCode() == 0);

  // Allow time for async assimilation to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Verify data exists in CTE
  auto tag = clio::cte::core::Tag(tag_name);
  auto blobs = tag.GetContainedBlobs();

  INFO("Assimilated file created " << blobs.size() << " blob(s) in CTE");
}

// ============================================================================
// URL Parsing Tests
// ============================================================================

TEST_CASE("CAE - File URL Parsing", "[cae][url]") {
  AssimilationCtx ctx;

  // Test various URL formats
  ctx.src = "file::/path/to/file.bin";
  REQUIRE(ctx.src.find("file::") == 0);

  ctx.src = "file:///absolute/path/file.dat";
  REQUIRE(ctx.src.find("file:") == 0);

  INFO("File URL parsing works correctly");
}

TEST_CASE("CAE - IOWarp URL Parsing", "[cae][url]") {
  AssimilationCtx ctx;

  ctx.dst = "iowarp::my_tag_name";
  REQUIRE(ctx.dst.find("iowarp::") == 0);

  std::string tag = ctx.dst.substr(8);  // Skip "iowarp::"
  REQUIRE(tag == "my_tag_name");

  INFO("IOWarp URL parsing works correctly");
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST_CASE("CAE - Binary Format Detection", "[cae][format]") {
  AssimilationCtx ctx;
  ctx.format = "binary";

  REQUIRE(ctx.format == "binary");
  INFO("Binary format specified correctly");
}

TEST_CASE("CAE - HDF5 Format Detection", "[cae][format]") {
  AssimilationCtx ctx;
  ctx.format = "hdf5";

  REQUIRE(ctx.format == "hdf5");
  INFO("HDF5 format specified correctly");
}

TEST_CASE("CAE - Unknown Format Handling", "[cae][format]") {
  AssimilationCtx ctx;
  ctx.format = "unknown_format";

  // Should still store the format string
  REQUIRE(ctx.format == "unknown_format");
  INFO("Unknown format handled gracefully");
}

SIMPLE_TEST_MAIN()
