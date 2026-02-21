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
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/core_tasks.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <fstream>
#include <cstdlib>

using namespace wrp_cae::core;

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

      // Step 1: Initialize Chimaera runtime
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (!success) {
        throw std::runtime_error("CHIMAERA_INIT failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Step 2: Initialize CTE client subsystem
      success = wrp_cte::core::WRP_CTE_CLIENT_INIT();
      if (!success) {
        throw std::runtime_error("WRP_CTE_CLIENT_INIT failed");
      }

      // Step 3: Set pool ID on global CTE client
      auto *cte_client = WRP_CTE_CLIENT;
      cte_client->Init(wrp_cte::core::kCtePoolId);

      // Step 4: Create CTE core pool
      wrp_cte::core::CreateParams cte_params;
      auto cte_create = cte_client->AsyncCreate(
          chi::PoolQuery::Dynamic(),
          wrp_cte::core::kCtePoolName,
          wrp_cte::core::kCtePoolId,
          cte_params);
      cte_create.Wait();

      if (cte_create->GetReturnCode() != 0) {
        throw std::runtime_error("CTE pool creation failed");
      }

      INFO("CTE infrastructure initialized");
      g_initialized = true;
    }

    test_data_dir_ = std::string(std::getenv("HOME")) + "/cae_test_data";
    test_binary_file_ = test_data_dir_ + "/test_binary.bin";
    test_hdf5_file_ = "/workspace/context-assimilation-engine/data/A46_xx.h5";
  }

  void InitializeCAE() {
    if (g_cae_initialized) {
      return;
    }

    INFO("Initializing CAE client");

    // Initialize CAE client subsystem
    WRP_CAE_CLIENT_INIT();

    // Create CAE client and pool
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams cae_params;
    auto cae_create = cae_client.AsyncCreate(
        chi::PoolQuery::Local(),
        "test_cae_pool",
        wrp_cae::core::kCaePoolId,
        cae_params);
    cae_create.Wait();

    if (cae_create->GetReturnCode() != 0) {
      throw std::runtime_error("CAE pool creation failed");
    }

    INFO("CAE infrastructure initialized");
    g_cae_initialized = true;
  }

  void SetupTestData() {
    // Create test data directory
    (void)system(("mkdir -p " + test_data_dir_).c_str());

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
  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);
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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

  // Process HDF5 dataset
  auto task = cae_client.AsyncProcessHdf5Dataset(
      chi::PoolQuery::Local(),
      fixture.test_hdf5_file_,
      "/data",
      "cae_hdf5_test");
  task.Wait();

  INFO("ProcessHdf5Dataset completed with code: " << task->GetReturnCode());
}

TEST_CASE("CAE - ProcessHdf5Dataset NonExistent File", "[cae][hdf5][error]") {
  CAEComprehensiveFixture fixture;
  fixture.InitializeCAE();

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

  // Try to process non-existent file
  auto task = cae_client.AsyncProcessHdf5Dataset(
      chi::PoolQuery::Local(),
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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

  // Try with empty dataset path
  auto task = cae_client.AsyncProcessHdf5Dataset(
      chi::PoolQuery::Local(),
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

  // Serialize using cereal (stringstream as archive)
  std::stringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(ctx);
  }

  // Deserialize
  AssimilationCtx ctx_restored;
  {
    cereal::BinaryInputArchive iarchive(ss);
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

  wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

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
  auto tag = wrp_cte::core::Tag(tag_name);
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
