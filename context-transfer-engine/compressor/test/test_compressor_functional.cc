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
 * Compressor ChiMod Functional Tests
 *
 * Tests the actual functionality of the compressor chimod tasks:
 * - CompressTask: Compression with various libraries
 * - DecompressTask: Decompression and data integrity
 * - DynamicScheduleTask: Intelligent compression selection
 * - Round-trip: Compress + Decompress data verification
 *
 * NOTE: The compressor API has been changed to integrate with the core module.
 * CompressTask now has the same inputs as PutBlobTask and calls PutBlob internally.
 * DecompressTask now has the same inputs as GetBlobTask and calls GetBlob internally.
 * These tests require a fully initialized CTE environment with core pool.
 */

#include "simple_test.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>

#include <chimaera/chimaera.h>
#include <wrp_cte/compressor/compressor_client.h>
#include <wrp_cte/compressor/compressor_tasks.h>
#include <wrp_cte/compressor/compressor_runtime.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

using namespace wrp_cte::compressor;

namespace {

// Compression library IDs
namespace CompLib {
  constexpr int NONE = 0;
  constexpr int BROTLI = 0;
  constexpr int BZIP2 = 1;
  constexpr int BLOSC2 = 2;
  constexpr int FPZIP = 3;
  constexpr int LZ4 = 4;
  constexpr int LZMA = 5;
  constexpr int SNAPPY = 6;
  constexpr int SZ3 = 7;
  constexpr int ZFP = 8;
  constexpr int ZLIB = 9;
  constexpr int ZSTD = 10;
}

/**
 * Generate test data with specified pattern
 */
std::vector<char> GenerateTestData(size_t size, const std::string& pattern) {
  std::vector<char> data(size);

  if (pattern == "zeros") {
    // All zeros - highly compressible
    std::fill(data.begin(), data.end(), 0);
  } else if (pattern == "ones") {
    // All ones - highly compressible
    std::fill(data.begin(), data.end(), 1);
  } else if (pattern == "repeating") {
    // Repeating pattern - moderately compressible
    const char pattern_bytes[] = {0x01, 0x02, 0x03, 0x04};
    for (size_t i = 0; i < size; ++i) {
      data[i] = pattern_bytes[i % 4];
    }
  } else if (pattern == "random") {
    // Random data - poorly compressible
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(dis(gen));
    }
  } else if (pattern == "text") {
    // Text-like data - moderately compressible
    const char* text = "The quick brown fox jumps over the lazy dog. ";
    size_t text_len = strlen(text);
    for (size_t i = 0; i < size; ++i) {
      data[i] = text[i % text_len];
    }
  }

  return data;
}

/**
 * Initialize Chimaera runtime for compressor tests
 */
void InitializeChimaera() {
  // Initialize Chimaera runtime in client mode with runtime
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!success) {
    throw std::runtime_error("Failed to initialize Chimaera runtime");
  }
}

/**
 * Cleanup Chimaera runtime
 */
void CleanupChimaera() {
  // Client finalize handled by CHI_CLIENT destructor
}

/**
 * Create and return pool ID for core chimod
 */
chi::PoolId CreateCorePool() {
  chi::PoolId core_pool_id = chi::PoolId(1, 1);
  wrp_cte::core::Client core_client;

  wrp_cte::core::CreateParams core_params;
  auto create_task = core_client.AsyncCreate(
      chi::PoolQuery::Local(),
      "test_core_pool",
      core_pool_id,
      core_params);
  create_task.Wait();

  return core_pool_id;
}

/**
 * Create and return pool ID for compressor chimod
 */
chi::PoolId CreateCompressorPool() {
  chi::PoolId compressor_pool_id = chi::PoolId(2, 1);
  Client compressor_client;

  auto create_task = compressor_client.AsyncCreate(
      chi::PoolQuery::Local(),
      "test_compressor_pool",
      compressor_pool_id);
  create_task.Wait();

  return compressor_pool_id;
}

/**
 * Test fixture for CTE integration tests
 */
struct CTETestFixture {
  chi::PoolId core_pool_id_;
  chi::PoolId compressor_pool_id_;
  wrp_cte::core::Client core_client_;
  Client compressor_client_;
  wrp_cte::core::TagId tag_id_;

  CTETestFixture() {
    InitializeChimaera();
    core_pool_id_ = CreateCorePool();
    compressor_pool_id_ = CreateCompressorPool();
    core_client_.Init(core_pool_id_);
    compressor_client_.Init(compressor_pool_id_);

    // Create a test tag
    auto tag_task = core_client_.AsyncGetOrCreateTag("test_tag");
    tag_task.Wait();
    tag_id_ = tag_task->tag_id_;
  }

  ~CTETestFixture() {
    CleanupChimaera();
  }

  /**
   * Allocate shared memory and copy data to it
   */
  hipc::FullPtr<char> AllocateAndCopyData(const std::vector<char>& data) {
    auto shm_buffer = CHI_IPC->AllocateBuffer(data.size());
    if (!shm_buffer.IsNull()) {
      std::memcpy(shm_buffer.ptr_, data.data(), data.size());
    }
    return shm_buffer;
  }

  /**
   * Read data from shared memory
   */
  std::vector<char> ReadFromSharedMemory(hipc::FullPtr<char>& buffer, size_t size) {
    std::vector<char> data(size);
    if (!buffer.IsNull()) {
      std::memcpy(data.data(), buffer.ptr_, size);
    }
    return data;
  }
};

}  // namespace

/**
 * Test Case 1: Basic Compress and Store via PutBlob
 * Tests that CompressTask properly compresses data and stores via core PutBlob
 */
TEST_CASE("Basic Compress and Store", "[compressor][functional][basic]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(16 * 1024, "text");

  // Allocate shared memory for input data
  auto shm_buffer = fixture.AllocateAndCopyData(test_data);
  REQUIRE(!shm_buffer.IsNull());

  hipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  // Call AsyncCompress which compresses and stores via PutBlob
  auto task = fixture.compressor_client_.AsyncCompress(
      chi::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_compress",
      0,  // offset
      test_data.size(),
      blob_data,
      0.5f,  // score
      context,
      0,  // flags
      fixture.core_pool_id_);
  task.Wait();

  REQUIRE(task->return_code_ == 0);
  INFO("Compression completed successfully");

  // Cleanup
  CHI_IPC->FreeBuffer(shm_buffer);
}

/**
 * Test Case 2: Decompress and Retrieve via GetBlob
 * Tests that DecompressTask properly retrieves and decompresses data
 */
TEST_CASE("Decompress and Retrieve", "[compressor][functional][basic]") {
  CTETestFixture fixture;

  auto original_data = GenerateTestData(16 * 1024, "text");

  // First, compress and store the data
  auto put_buffer = fixture.AllocateAndCopyData(original_data);
  REQUIRE(!put_buffer.IsNull());

  hipc::ShmPtr<> put_blob_data = put_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = CompLib::LZ4;
  context.compress_preset_ = 2;

  auto compress_task = fixture.compressor_client_.AsyncCompress(
      chi::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_roundtrip",
      0,
      original_data.size(),
      put_blob_data,
      0.5f,
      context,
      0,
      fixture.core_pool_id_);
  compress_task.Wait();
  REQUIRE(compress_task->return_code_ == 0);

  CHI_IPC->FreeBuffer(put_buffer);

  // Now retrieve and decompress
  auto get_buffer = CHI_IPC->AllocateBuffer(original_data.size());
  REQUIRE(!get_buffer.IsNull());

  hipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

  auto decompress_task = fixture.compressor_client_.AsyncDecompress(
      chi::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_roundtrip",
      0,
      original_data.size(),
      0,  // flags
      get_blob_data,
      fixture.core_pool_id_);
  decompress_task.Wait();

  REQUIRE(decompress_task->return_code_ == 0);
  REQUIRE(decompress_task->output_size_ == original_data.size());

  // Verify data integrity
  auto retrieved_data = fixture.ReadFromSharedMemory(get_buffer, original_data.size());
  REQUIRE(std::memcmp(original_data.data(), retrieved_data.data(), original_data.size()) == 0);

  INFO("Round-trip compression/decompression verified");
  CHI_IPC->FreeBuffer(get_buffer);
}

/**
 * Test Case 3: Dynamic Schedule
 * Tests that DynamicScheduleTask selects optimal compression
 */
TEST_CASE("Dynamic Schedule Compression", "[compressor][functional][dynamic]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(64 * 1024, "text");

  auto shm_buffer = fixture.AllocateAndCopyData(test_data);
  REQUIRE(!shm_buffer.IsNull());

  hipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

  Context context;
  context.dynamic_compress_ = 0;  // Enable dynamic compression selection
  context.max_performance_ = false;  // Optimize for ratio

  auto task = fixture.compressor_client_.AsyncDynamicSchedule(
      chi::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_dynamic",
      0,
      test_data.size(),
      blob_data,
      0.5f,
      context,
      0,
      fixture.core_pool_id_);
  task.Wait();

  REQUIRE(task->return_code_ == 0);
  INFO("DynamicSchedule selected compression library: " << task->context_.compress_lib_);
  INFO("Tier score: " << task->tier_score_);

  CHI_IPC->FreeBuffer(shm_buffer);
}

/**
 * Test Case 4: Multiple Compression Libraries
 * Tests compression with various libraries
 */
TEST_CASE("Multiple Compression Libraries", "[compressor][functional][libraries]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(16 * 1024, "text");

  std::vector<std::pair<int, std::string>> libraries = {
    {CompLib::LZ4, "LZ4"},
    {CompLib::ZSTD, "ZSTD"},
    {CompLib::ZLIB, "ZLIB"}
  };

  for (const auto& [lib_id, lib_name] : libraries) {
    SECTION(lib_name) {
      auto shm_buffer = fixture.AllocateAndCopyData(test_data);
      REQUIRE(!shm_buffer.IsNull());

      hipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

      Context context;
      context.compress_lib_ = lib_id;
      context.compress_preset_ = 2;

      std::string blob_name = "test_blob_" + lib_name;
      auto task = fixture.compressor_client_.AsyncCompress(
          chi::PoolQuery::Local(),
          fixture.tag_id_,
          blob_name,
          0,
          test_data.size(),
          blob_data,
          0.5f,
          context,
          0,
          fixture.core_pool_id_);
      task.Wait();

      REQUIRE(task->return_code_ == 0);
      INFO(lib_name << " compression completed successfully");

      CHI_IPC->FreeBuffer(shm_buffer);
    }
  }
}

/**
 * Test Case 5: No Compression (Passthrough)
 * Tests that data with compress_lib_ = 0 is stored without compression
 */
TEST_CASE("No Compression Passthrough", "[compressor][functional][passthrough]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(8 * 1024, "random");

  auto shm_buffer = fixture.AllocateAndCopyData(test_data);
  REQUIRE(!shm_buffer.IsNull());

  hipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

  Context context;
  context.compress_lib_ = 0;  // No compression

  auto task = fixture.compressor_client_.AsyncCompress(
      chi::PoolQuery::Local(),
      fixture.tag_id_,
      "test_blob_passthrough",
      0,
      test_data.size(),
      blob_data,
      0.5f,
      context,
      0,
      fixture.core_pool_id_);
  task.Wait();

  REQUIRE(task->return_code_ == 0);
  INFO("Passthrough (no compression) completed successfully");

  CHI_IPC->FreeBuffer(shm_buffer);
}

/**
 * Test Case 6: Error Handling - Invalid Parameters
 */
TEST_CASE("Error Handling - Invalid Parameters", "[compressor][functional][error]") {
  CTETestFixture fixture;

  auto test_data = GenerateTestData(1024, "text");

  SECTION("Null blob data") {
    Context context;
    context.compress_lib_ = CompLib::LZ4;

    auto task = fixture.compressor_client_.AsyncCompress(
        chi::PoolQuery::Local(),
        fixture.tag_id_,
        "test_blob_null",
        0,
        test_data.size(),
        hipc::ShmPtr<>::GetNull(),  // Null data
        0.5f,
        context,
        0,
        fixture.core_pool_id_);
    task.Wait();

    // Should fail gracefully
    REQUIRE(task->return_code_ != 0);
    INFO("Correctly handled null blob data");
  }

  SECTION("Zero size") {
    auto shm_buffer = fixture.AllocateAndCopyData(test_data);
    REQUIRE(!shm_buffer.IsNull());

    hipc::ShmPtr<> blob_data = shm_buffer.shm_.template Cast<void>();

    Context context;
    context.compress_lib_ = CompLib::LZ4;

    auto task = fixture.compressor_client_.AsyncCompress(
        chi::PoolQuery::Local(),
        fixture.tag_id_,
        "test_blob_zero_size",
        0,
        0,  // Zero size
        blob_data,
        0.5f,
        context,
        0,
        fixture.core_pool_id_);
    task.Wait();

    // Should fail gracefully
    REQUIRE(task->return_code_ != 0);
    INFO("Correctly handled zero size");

    CHI_IPC->FreeBuffer(shm_buffer);
  }
}

// Main function using simple_test.h framework
SIMPLE_TEST_MAIN()
