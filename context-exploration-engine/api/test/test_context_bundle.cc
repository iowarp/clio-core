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
 * test_context_bundle.cc - Unit test for ContextInterface bundle and query APIs
 *
 * This test validates the ContextInterface API by:
 * 1. Creating a test binary file with patterned data
 * 2. Bundling it using ContextBundle with AssimilationCtx
 * 3. Querying CTE to verify the data was ingested
 * 4. Validating the complete bundle-and-retrieve workflow
 *
 * Test Strategy:
 * - Tests empty bundle handling (edge case)
 * - Tests AssimilationCtx constructor
 * - Tests real data bundling workflow (integration test)
 * - Tests query functionality after bundling
 *
 * Environment Variables:
 * - INIT_CHIMAERA: If set to "1", initializes Chimaera runtime
 */

#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <hermes_shm/util/logging.h>

// Test configuration
const std::string kTestFileName = "/tmp/test_cee_bundle_file.bin";
const std::string kTestTagName = "cee_test_bundle_tag";
const size_t kTestFileSize = 1024 * 1024;  // 1 MB test file

/**
 * Generate a test file with patterned data for validation
 * Pattern: Each 4-byte block contains the block index (little endian)
 */
bool GenerateTestFile(const std::string& file_path, size_t size_bytes) {
  HLOG(kInfo, "Generating test file: {} ({} bytes)", file_path, size_bytes);

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    HLOG(kError, "Failed to create test file: {}", file_path);
    return false;
  }

  // Generate patterned data (4-byte blocks with incrementing indices)
  const size_t block_size = 4;
  const size_t num_blocks = size_bytes / block_size;

  for (size_t i = 0; i < num_blocks; ++i) {
    uint32_t value = static_cast<uint32_t>(i);
    file.write(reinterpret_cast<const char*>(&value), block_size);

    if (!file.good()) {
      HLOG(kError, "Failed to write to test file at block {}", i);
      file.close();
      return false;
    }
  }

  file.close();
  HLOG(kSuccess, "Test file generated successfully ({} blocks)", num_blocks);
  return true;
}

/**
 * Test that context_bundle can handle empty bundles
 */
void test_empty_bundle() {
  HLOG(kInfo, "TEST: Empty bundle");

  iowarp::ContextInterface ctx_interface;
  std::vector<wrp_cae::core::AssimilationCtx> empty_bundle;

  // Empty bundle should return success (0)
  int result = ctx_interface.ContextBundle(empty_bundle);
  assert(result == 0 && "Empty bundle should return success");

  HLOG(kSuccess, "PASSED: Empty bundle test");
}

/**
 * Test AssimilationCtx constructor with all parameters
 */
void test_assimilation_ctx_constructor() {
  HLOG(kInfo, "TEST: AssimilationCtx constructor");

  wrp_cae::core::AssimilationCtx ctx(
      "file::/path/to/source.dat",
      "iowarp::dest_tag",
      "binary",
      "dependency_id",
      1024,
      2048,
      "src_access_token",
      "dst_access_token");

  assert(ctx.src == "file::/path/to/source.dat");
  assert(ctx.dst == "iowarp::dest_tag");
  assert(ctx.format == "binary");
  assert(ctx.depends_on == "dependency_id");
  assert(ctx.range_off == 1024);
  assert(ctx.range_size == 2048);
  assert(ctx.src_token == "src_access_token");
  assert(ctx.dst_token == "dst_access_token");

  HLOG(kSuccess, "PASSED: AssimilationCtx constructor test");
}

/**
 * Test comprehensive bundle-and-retrieve workflow
 * This is the main integration test that validates:
 * 1. File creation
 * 2. Bundle operation (assimilation)
 * 3. Query operation (verification)
 */
void test_bundle_and_retrieve_workflow() {
  HLOG(kInfo, "TEST: Bundle-and-retrieve workflow");

  // Step 1: Generate test file
  HLOG(kInfo, "[STEP 1] Generating test file...");
  if (!GenerateTestFile(kTestFileName, kTestFileSize)) {
    assert(false && "Failed to generate test file");
  }

  // Step 2: Initialize CTE client
  HLOG(kInfo, "[STEP 2] Initializing CTE client...");
  wrp_cte::core::WRP_CTE_CLIENT_INIT();

  // Step 2.5: Register a RAM storage target with CTE
  HLOG(kInfo, "[STEP 2.5] Registering RAM storage target with CTE...");
  auto* cte_client = WRP_CTE_CLIENT;
  auto register_task = cte_client->AsyncRegisterTarget(
      "ram::cee_test_storage",  // Target name (RAM storage)
      chimaera::bdev::BdevType::kRam,  // RAM block device type
      4ULL * 1024 * 1024 * 1024,  // 4GB capacity
      chi::PoolQuery::Local(),  // Local pool query for single-node
      chi::PoolId(800, 0));  // Explicit bdev pool ID
  register_task.Wait();
  chi::u32 register_result = register_task->return_code_;
  assert(register_result == 0 && "Failed to register storage target");
  HLOG(kSuccess, "Storage target registered successfully");

  // Step 3: Create CAE pool
  HLOG(kInfo, "[STEP 3] Creating CAE pool...");
  wrp_cae::core::Client cae_client;
  wrp_cae::core::CreateParams params;

  auto create_task = cae_client.AsyncCreate(
      chi::PoolQuery::Local(),
      "test_cee_cae_pool",
      wrp_cae::core::kCaePoolId,
      params);
  create_task.Wait();

  HLOG(kSuccess, "CAE pool created with ID: {}", cae_client.pool_id_);

  // Step 4: Bundle the test file using ContextInterface
  HLOG(kInfo, "[STEP 4] Bundling test file...");
  iowarp::ContextInterface ctx_interface;

  std::vector<wrp_cae::core::AssimilationCtx> bundle;
  wrp_cae::core::AssimilationCtx ctx;
  ctx.src = "file::" + kTestFileName;
  ctx.dst = "iowarp::" + kTestTagName;
  ctx.format = "binary";
  ctx.depends_on = "";
  ctx.range_off = 0;
  ctx.range_size = 0;  // 0 means full file
  ctx.src_token = "";
  ctx.dst_token = "";
  bundle.push_back(ctx);

  int bundle_result = ctx_interface.ContextBundle(bundle);
  assert(bundle_result == 0 && "ContextBundle should return success");
  HLOG(kSuccess, "Bundle operation completed successfully");

  // Step 5: Wait for assimilation to complete
  HLOG(kInfo, "[STEP 5] Waiting for assimilation to complete...");
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Step 6: Query to verify the data was ingested
  HLOG(kInfo, "[STEP 6] Querying for bundled data...");
  std::vector<std::string> query_results = ctx_interface.ContextQuery(
      kTestTagName,  // Exact tag name
      ".*");         // Match all blobs

  HLOG(kInfo, "Query returned {} results", query_results.size());

  // Verify we got at least one result
  if (query_results.size() > 0) {
    HLOG(kInfo, "Found blobs in tag '{}':", kTestTagName);
    for (const auto& blob : query_results) {
      HLOG(kInfo, "  - {}", blob);
    }
  }

  // Note: The number of blobs depends on chunking behavior
  // For a 1MB file, it might be stored as multiple blobs
  assert(query_results.size() > 0 && "Query should find at least one blob after bundling");

  // Step 7: Cleanup - destroy the context
  HLOG(kInfo, "[STEP 7] Cleaning up test context...");
  std::vector<std::string> contexts_to_delete = {kTestTagName};
  int destroy_result = ctx_interface.ContextDestroy(contexts_to_delete);
  HLOG(kInfo, "Destroy returned code: {}", destroy_result);

  // Step 8: Delete test file
  std::remove(kTestFileName.c_str());

  HLOG(kSuccess, "PASSED: Bundle-and-retrieve workflow test");
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  HLOG(kInfo, "========================================");
  HLOG(kInfo, "ContextInterface::ContextBundle Tests");
  HLOG(kInfo, "========================================");

  try {
    // Initialize Chimaera runtime if requested (for unit tests)
    const char* init_chimaera = std::getenv("INIT_CHIMAERA");
    if (init_chimaera && std::strcmp(init_chimaera, "1") == 0) {
      HLOG(kInfo, "Initializing Chimaera (INIT_CHIMAERA=1)...");
      chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      HLOG(kSuccess, "Chimaera initialized");
    }

    // Verify Chimaera IPC is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Chimaera IPC not initialized. Is the runtime running?");
      HLOG(kInfo, "HINT: Set INIT_CHIMAERA=1 to initialize runtime or start runtime externally");
      return 1;
    }
    HLOG(kSuccess, "Chimaera IPC verified");

    // Run all tests
    test_empty_bundle();

    test_assimilation_ctx_constructor();

    test_bundle_and_retrieve_workflow();

    HLOG(kSuccess, "All tests PASSED!");
    return 0;
  } catch (const std::exception& e) {
    HLOG(kError, "Test FAILED with exception: {}", e.what());
    return 1;
  }
}
