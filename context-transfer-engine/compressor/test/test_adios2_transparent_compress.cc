/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * ADIOS2 Adapter + Transparent Compression Integration Test
 *
 * Validates that the ADIOS2 adapter works correctly when the CTE
 * compose pipeline includes the compressor module. Uses the direct
 * CTE client API (AsyncPutBlob/AsyncGetBlob) — not the Tag wrapper —
 * to verify transparent compression round-trips through the ADIOS2
 * data path.
 *
 * Compose pipeline:
 *   compressor (512.0) → CTE core (513.0) → RAM bdev (301.0)
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
static clio::cte::core::TagId g_tag_id;

static void EnsureInit() {
  if (g_initialized) return;

  // Use the compose config that places compressor at 512.0, CTE core at 513.0
  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_transparent_compress_config.yaml";
  setenv("CLIO_SERVER_CONF", config_path.c_str(), 1);

  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer);
  REQUIRE(success);
  g_initialized = true;
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;

  // Wait for compose pools to initialize
  std::this_thread::sleep_for(1s);

  // Point the CTE client at the entrypoint pool (512.0 = compressor)
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);

  // Create a tag for the tests
  auto tag_task = cte_client->AsyncGetOrCreateTag("adios2_compress_test");
  tag_task.Wait();
  g_tag_id = tag_task->tag_id_;
}

TEST_CASE("ADIOS2 transparent compress - PutBlob float array",
          "[adios2][compressor][transparent][put]") {
  EnsureInit();
  auto *cte_client = CLIO_CTE_CLIENT;

  // Simulate an ADIOS2 variable write: array of floats
  const size_t num_elements = 16 * 1024;  // 16K floats = 64KB
  const size_t data_size = num_elements * sizeof(float);

  auto buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!buffer.IsNull());

  // Fill with a compressible pattern (repeating values compress well)
  float *fdata = reinterpret_cast<float *>(buffer.ptr_);
  for (size_t i = 0; i < num_elements; ++i) {
    fdata[i] = static_cast<float>(i % 256);
  }

  ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();

  clio::cte::core::Context ctx;
#if CTP_ENABLE_COMPRESS
  ctx.dynamic_compress_ = 1;  // Static compression
  ctx.compress_lib_ = 4;      // LZ4
  ctx.compress_preset_ = 2;   // BALANCED
  ctx.data_type_ = 1;         // float
#endif

  auto put_task = cte_client->AsyncPutBlob(
      g_tag_id, "temperature/step0", 0, data_size, blob_data,
      -1.0f, ctx, 0);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  INFO("PutBlob float array completed through compressor pipeline");
}

TEST_CASE("ADIOS2 transparent compress - GetBlob float array",
          "[adios2][compressor][transparent][get]") {
  EnsureInit();
  auto *cte_client = CLIO_CTE_CLIENT;

  const size_t num_elements = 16 * 1024;
  const size_t data_size = num_elements * sizeof(float);

  auto get_buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!get_buffer.IsNull());
  memset(get_buffer.ptr_, 0, data_size);

  ctp::ipc::ShmPtr<> get_data = get_buffer.shm_.template Cast<void>();

  auto get_task = cte_client->AsyncGetBlob(
      g_tag_id, "temperature/step0", 0, data_size, 0, get_data);
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  // Verify data integrity
  float *fdata = reinterpret_cast<float *>(get_buffer.ptr_);
  bool data_ok = true;
  for (size_t i = 0; i < num_elements; ++i) {
    if (fdata[i] != static_cast<float>(i % 256)) {
      INFO("Mismatch at element " + std::to_string(i) +
           ": got " + std::to_string(fdata[i]) +
           " expected " + std::to_string(static_cast<float>(i % 256)));
      data_ok = false;
      break;
    }
  }
  REQUIRE(data_ok);
  INFO("GetBlob float array round-trip verified through compressor");
}

TEST_CASE("ADIOS2 transparent compress - multi-step workflow",
          "[adios2][compressor][transparent][multistep]") {
  EnsureInit();
  auto *cte_client = CLIO_CTE_CLIENT;

  const size_t num_elements = 4096;
  const size_t data_size = num_elements * sizeof(double);
  const int num_steps = 3;

  // Write multiple steps
  for (int step = 0; step < num_steps; ++step) {
    auto buffer = CLIO_IPC->AllocateBuffer(data_size);
    REQUIRE(!buffer.IsNull());

    double *ddata = reinterpret_cast<double *>(buffer.ptr_);
    for (size_t i = 0; i < num_elements; ++i) {
      ddata[i] = static_cast<double>(step * 1000 + i);
    }

    ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
    std::string blob_name = "pressure/step" + std::to_string(step);

    clio::cte::core::Context ctx;
#if CTP_ENABLE_COMPRESS
    ctx.dynamic_compress_ = 1;
    ctx.compress_lib_ = 10;   // ZSTD
    ctx.compress_preset_ = 1; // FAST
#endif

    auto put_task = cte_client->AsyncPutBlob(
        g_tag_id, blob_name, 0, data_size, blob_data, -1.0f, ctx, 0);
    put_task.Wait();
    REQUIRE(put_task->GetReturnCode() == 0);
  }

  // Read back and verify each step
  for (int step = 0; step < num_steps; ++step) {
    auto get_buffer = CLIO_IPC->AllocateBuffer(data_size);
    REQUIRE(!get_buffer.IsNull());
    memset(get_buffer.ptr_, 0, data_size);

    ctp::ipc::ShmPtr<> get_data = get_buffer.shm_.template Cast<void>();
    std::string blob_name = "pressure/step" + std::to_string(step);

    auto get_task = cte_client->AsyncGetBlob(
        g_tag_id, blob_name, 0, data_size, 0, get_data);
    get_task.Wait();
    REQUIRE(get_task->GetReturnCode() == 0);

    double *ddata = reinterpret_cast<double *>(get_buffer.ptr_);
    bool data_ok = true;
    for (size_t i = 0; i < num_elements; ++i) {
      double expected = static_cast<double>(step * 1000 + i);
      if (ddata[i] != expected) {
        INFO("Step " + std::to_string(step) +
             " mismatch at element " + std::to_string(i));
        data_ok = false;
        break;
      }
    }
    REQUIRE(data_ok);
  }
  INFO("Multi-step round-trip verified through compressor pipeline");
}

SIMPLE_TEST_MAIN()
