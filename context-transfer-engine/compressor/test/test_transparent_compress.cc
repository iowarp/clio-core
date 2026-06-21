/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Transparent Compression Integration Test
 *
 * Validates that CLIO_CTE_CLIENT_INIT() with a compose config placing the
 * compressor at pool 512.0 (the CTE entrypoint) and CTE core at 513.0
 * results in transparent compression when calling the standard CTE client
 * AsyncPutBlob / AsyncGetBlob API.
 */

#include "simple_test.h"
#include <cstring>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>

using namespace std::chrono_literals;
namespace fs = std::filesystem;

static bool g_initialized = false;

static void EnsureInit() {
  if (g_initialized) return;

  // Point CLIO Runtime at our compose config with compressor at 512.0
  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_transparent_compress_config.yaml";
  setenv("CLIO_SERVER_CONF", config_path.c_str(), 1);

  // Start as server — compose will create all pools (compressor at 512.0,
  // CTE core at 513.0, bdev at 301.0).
  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer);
  REQUIRE(success);
  g_initialized = true;
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;

  // Wait for compose pools to initialize
  std::this_thread::sleep_for(1s);

  // Point the global CTE client at the entrypoint pool (512.0 = compressor).
  // Don't call CLIO_CTE_CLIENT_INIT — compose already created everything.
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);
}

TEST_CASE("Transparent PutBlob through compressor",
          "[compressor][transparent][putblob]") {
  EnsureInit();
  auto *cte_client = CLIO_CTE_CLIENT;

  // Create a tag
  auto tag_task = cte_client->AsyncGetOrCreateTag("transparent_test_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  // Create compressible data (repeating pattern compresses well)
  const size_t data_size = 64 * 1024;  // 64 KB
  auto buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!buffer.IsNull());
  char *data_ptr = buffer.ptr_;
  const char pattern[] = "ABCDEFGH";
  for (size_t i = 0; i < data_size; ++i) {
    data_ptr[i] = pattern[i % sizeof(pattern)];
  }

  ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();

  // PutBlob — this should go through the compressor (512.0) transparently
  clio::cte::core::Context ctx;
#if CTP_ENABLE_COMPRESS
  ctx.dynamic_compress_ = 1;  // Enable static compression
  ctx.compress_lib_ = 4;      // LZ4
  ctx.compress_preset_ = 2;   // BALANCED
#endif

  auto put_task = cte_client->AsyncPutBlob(
      tag_id, "test_blob_transparent", 0, data_size, blob_data,
      0.5f, ctx, 0);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  INFO("PutBlob completed successfully");
}

TEST_CASE("Transparent GetBlob through compressor",
          "[compressor][transparent][getblob]") {
  EnsureInit();
  auto *cte_client = CLIO_CTE_CLIENT;

  // Get the tag we created in the previous test
  auto tag_task = cte_client->AsyncGetOrCreateTag("transparent_test_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  // Allocate receive buffer
  const size_t data_size = 64 * 1024;
  auto get_buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!get_buffer.IsNull());
  memset(get_buffer.ptr_, 0, data_size);

  ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

  // GetBlob — should retrieve and decompress transparently
  auto get_task = cte_client->AsyncGetBlob(
      tag_id, "test_blob_transparent", 0, data_size, 0, get_blob_data);
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  // Verify data integrity
  const char pattern[] = "ABCDEFGH";
  bool data_matches = true;
  for (size_t i = 0; i < data_size; ++i) {
    if (get_buffer.ptr_[i] != pattern[i % sizeof(pattern)]) {
      data_matches = false;
      INFO("Data mismatch at byte " + std::to_string(i));
      break;
    }
  }
  REQUIRE(data_matches);
  INFO("GetBlob returned correct data - round-trip verified");
}

SIMPLE_TEST_MAIN()
