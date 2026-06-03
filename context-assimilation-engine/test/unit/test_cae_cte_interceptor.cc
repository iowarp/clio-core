/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CAE -> CTE Interceptor Integration Test
 *
 * Validates that a compose config placing the CAE core at pool 512.0
 * (the CTE entrypoint) with next_pool_id 513.0 results in transparent
 * forwarding when calling the standard CTE client AsyncGetOrCreateTag /
 * AsyncPutBlob / AsyncGetBlob API. The CAE core acts as a passthrough
 * interceptor that forwards each task to the real CTE core at 513.0.
 *
 * This is the plumbing milestone for the broader transparent data
 * labeling work (see GitHub issue #467, umbrella #466). No labeling is
 * performed in CAE yet — these handlers just forward.
 *
 * The full GetOrCreateTag → PutBlob → GetBlob round-trip runs inside a
 * single TEST_CASE so that all calls share one chimaera server lifetime
 * (the RAM bdev storage doesn't survive a process restart).
 */

#include "simple_test.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_ctp/introspect/system_info.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

TEST_CASE("CLIO_CTE_CLIENT PutBlob writes to CAE interceptor and reads back",
          "[cae][interceptor][putblob][getblob]") {
  fs::path config_path = fs::path(__FILE__).parent_path() /
                          "test_cae_cte_interceptor_config.yaml";
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path.string(), 1);

  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer);
  REQUIRE(success);
  SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;

  // Give compose pools a moment to materialize.
  std::this_thread::sleep_for(1s);

  // Point the global CTE client at the entrypoint pool (512.0 = CAE
  // interceptor). All inbound AsyncPutBlob / AsyncGetBlob /
  // AsyncGetOrCreateTag tasks land in the CAE container and CAE forwards
  // them to the CTE container at 513.0.
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);

  // ---- GetOrCreateTag through CAE -----------------------------------
  auto tag_task = cte_client->AsyncGetOrCreateTag("cae_interceptor_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  REQUIRE(!tag_task->tag_id_.IsNull());
  auto tag_id = tag_task->tag_id_;

  // ---- PutBlob through CAE ------------------------------------------
  const size_t data_size = 16 * 1024;  // 16 KB
  auto buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!buffer.IsNull());
  for (size_t i = 0; i < data_size; ++i) {
    buffer.ptr_[i] = static_cast<char>('A' + (i % 26));
  }
  ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();

  clio::cte::core::Context ctx;
  auto put_task = cte_client->AsyncPutBlob(
      tag_id, "cae_interceptor_blob", 0, data_size, blob_data,
      0.5f, ctx, 0);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  // ---- GetBlob through CAE and verify -------------------------------
  auto get_buffer = CLIO_IPC->AllocateBuffer(data_size);
  REQUIRE(!get_buffer.IsNull());
  std::memset(get_buffer.ptr_, 0, data_size);
  ctp::ipc::ShmPtr<> get_blob_data = get_buffer.shm_.template Cast<void>();

  auto get_task = cte_client->AsyncGetBlob(
      tag_id, "cae_interceptor_blob", 0, data_size, 0, get_blob_data);
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  for (size_t i = 0; i < data_size; ++i) {
    char expected = static_cast<char>('A' + (i % 26));
    if (get_buffer.ptr_[i] != expected) {
      INFO("Mismatch at byte " + std::to_string(i) + ": expected " +
           std::to_string(static_cast<int>(expected)) + " got " +
           std::to_string(static_cast<int>(get_buffer.ptr_[i])));
      REQUIRE(get_buffer.ptr_[i] == expected);
    }
  }

  CLIO_IPC->FreeBuffer(buffer);
  CLIO_IPC->FreeBuffer(get_buffer);
}

SIMPLE_TEST_MAIN()
