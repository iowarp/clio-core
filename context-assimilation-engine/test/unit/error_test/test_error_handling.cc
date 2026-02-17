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
 * test_error_handling.cc - Unit test for ParseOmni API error handling
 *
 * This test validates that ParseOmni properly handles error conditions:
 * - Non-existent source file
 * - Invalid source protocol
 * - Invalid destination protocol
 * - Invalid range (out of bounds)
 * - Corrupted serialization
 *
 * Test Strategy:
 * - Negative testing: verify proper error codes for invalid inputs
 * - Boundary testing: verify range validation
 * - Protocol testing: verify protocol validation
 *
 * Expected behavior: All tests should FAIL gracefully with appropriate error codes
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <memory>

// Chimaera and CAE headers
#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

// CTE headers
#include <wrp_cte/core/core_client.h>

// Logging
#include <hermes_shm/util/logging.h>

// Bdev headers for storage target registration
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <filesystem>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

// Storage configuration
const std::string kTestStoragePath = "/tmp/cae_error_test_storage.dat";
const chi::u64 kTestTargetSize = 100 * 1024 * 1024;  // 100MB

// Test configuration
const std::string kTestFileName = "/tmp/test_error_handling_file.bin";
const std::string kNonExistentFile = "/tmp/nonexistent_file_12345.bin";

/**
 * Generate a small test file
 */
bool GenerateTestFile(const std::string& file_path, size_t size_bytes) {
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  std::vector<char> data(size_bytes, static_cast<char>(0xAB));
  file.write(data.data(), size_bytes);
  file.close();
  return true;
}

/**
 * Test error case - should fail with specific error code
 */
bool TestErrorCase(wrp_cae::core::Client& cae_client,
                   const std::string& test_name,
                   const wrp_cae::core::AssimilationCtx& ctx,
                   bool should_fail = true) {
  HLOG(kInfo, "--- Testing: {} ---", test_name);

  // Call ParseOmni with vector containing single context
  std::vector<wrp_cae::core::AssimilationCtx> contexts = {ctx};
  auto parse_task = cae_client.AsyncParseOmni(contexts);
  parse_task.Wait();
  // Use result_code_ (operation result) not GetReturnCode() (task completion status)
  chi::u32 result_code = parse_task->result_code_;
  chi::u32 num_tasks_scheduled = parse_task->num_tasks_scheduled_;

  HLOG(kInfo, "ParseOmni result: result_code={}, num_tasks={}", result_code, num_tasks_scheduled);

  // Validate
  if (should_fail) {
    // We expect this to fail
    if (result_code != 0) {
      HLOG(kSuccess, "{} failed as expected (error code: {})", test_name, result_code);
      return true;
    } else {
      HLOG(kError, "{} should have failed but succeeded", test_name);
      return false;
    }
  } else {
    // We expect this to succeed
    if (result_code == 0) {
      HLOG(kSuccess, "{} succeeded as expected", test_name);
      return true;
    } else {
      HLOG(kError, "{} should have succeeded but failed (error code: {})", test_name, result_code);
      return false;
    }
  }
}

/**
 * Note: TestCorruptedSerialization removed - no longer applicable
 * since ParseOmni now accepts AssimilationCtx directly and serialization
 * happens transparently in the task constructor.
 */

/**
 * Main test function
 */
int main(int argc, char* argv[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Error Handling ParseOmni Unit Test");
  HLOG(kInfo, "========================================");

  int exit_code = 0;
  int tests_passed = 0;
  int tests_total = 0;

  try {
    // Initialize Chimaera runtime (CHIMAERA_WITH_RUNTIME controls behavior)
    HLOG(kInfo, "Initializing Chimaera...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      HLOG(kError, "Failed to initialize Chimaera");
      return 1;
    }
    HLOG(kSuccess, "Chimaera initialized successfully");

    // Verify Chimaera IPC
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Chimaera IPC not initialized");
      return 1;
    }
    HLOG(kSuccess, "Chimaera IPC verified");

    // Generate test file (1MB)
    HLOG(kInfo, "[SETUP] Generating test file...");
    const size_t file_size = 1024 * 1024;
    if (!GenerateTestFile(kTestFileName, file_size)) {
      HLOG(kError, "Failed to generate test file");
      return 1;
    }

    // Connect to CTE
    HLOG(kInfo, "[SETUP] Connecting to CTE...");
    wrp_cte::core::WRP_CTE_CLIENT_INIT();

    // Set up storage target for CTE
    HLOG(kInfo, "[SETUP] Registering storage target...");

    // Clean up any existing test storage file
    if (fs::exists(kTestStoragePath)) {
      fs::remove(kTestStoragePath);
    }

    // Create bdev storage target
    chi::PoolId bdev_pool_id(200, 0);
    chimaera::bdev::Client bdev_client(bdev_pool_id);
    auto bdev_create_task = bdev_client.AsyncCreate(chi::PoolQuery::Dynamic(), kTestStoragePath,
                                                     bdev_pool_id, chimaera::bdev::BdevType::kFile);
    bdev_create_task.Wait();
    std::this_thread::sleep_for(100ms);

    // Register storage target with CTE
    auto *cte_client = WRP_CTE_CLIENT;
    auto reg_task = cte_client->AsyncRegisterTarget(kTestStoragePath,
                                                     chimaera::bdev::BdevType::kFile,
                                                     kTestTargetSize, chi::PoolQuery::Local(), bdev_pool_id);
    reg_task.Wait();
    std::this_thread::sleep_for(100ms);
    HLOG(kSuccess, "Storage target registered: {}", kTestStoragePath);

    // Initialize CAE client
    HLOG(kInfo, "[SETUP] Initializing CAE client...");
    WRP_CAE_CLIENT_INIT();

    // Create CAE pool
    HLOG(kInfo, "[SETUP] Creating CAE pool...");
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams params;

    auto create_task = cae_client.AsyncCreate(
        chi::PoolQuery::Local(),
        "test_cae_error_pool",
        wrp_cae::core::kCaePoolId,
        params);
    create_task.Wait();

    HLOG(kSuccess, "CAE pool created");

    // Test 1: Non-existent source file
    tests_total++;
    wrp_cae::core::AssimilationCtx ctx1;
    ctx1.src = "file::" + kNonExistentFile;
    ctx1.dst = "iowarp::test_error_tag1";
    ctx1.format = "binary";
    ctx1.range_off = 0;
    ctx1.range_size = 0;
    if (TestErrorCase(cae_client, "NonExistentFile", ctx1, true)) {
      tests_passed++;
    }

    // Test 2: Invalid source protocol
    tests_total++;
    wrp_cae::core::AssimilationCtx ctx2;
    ctx2.src = "invalid_protocol::/tmp/somefile.bin";
    ctx2.dst = "iowarp::test_error_tag2";
    ctx2.format = "binary";
    ctx2.range_off = 0;
    ctx2.range_size = 0;
    if (TestErrorCase(cae_client, "InvalidSourceProtocol", ctx2, true)) {
      tests_passed++;
    }

    // Test 3: Invalid destination protocol
    tests_total++;
    wrp_cae::core::AssimilationCtx ctx3;
    ctx3.src = "file::" + kTestFileName;
    ctx3.dst = "invalid_protocol::test_tag";
    ctx3.format = "binary";
    ctx3.range_off = 0;
    ctx3.range_size = 0;
    if (TestErrorCase(cae_client, "InvalidDestinationProtocol", ctx3, true)) {
      tests_passed++;
    }

    // Test 4: Out-of-range offset
    tests_total++;
    wrp_cae::core::AssimilationCtx ctx4;
    ctx4.src = "file::" + kTestFileName;
    ctx4.dst = "iowarp::test_error_tag4";
    ctx4.format = "binary";
    ctx4.range_off = file_size + 1000;  // Beyond file size
    ctx4.range_size = 100;
    if (TestErrorCase(cae_client, "OutOfRangeOffset", ctx4, true)) {
      tests_passed++;
    }

    // Test 5: Range size exceeds file
    tests_total++;
    wrp_cae::core::AssimilationCtx ctx5;
    ctx5.src = "file::" + kTestFileName;
    ctx5.dst = "iowarp::test_error_tag5";
    ctx5.format = "binary";
    ctx5.range_off = file_size - 100;
    ctx5.range_size = 1000;  // Would go past end of file
    if (TestErrorCase(cae_client, "RangeSizeExceedsFile", ctx5, true)) {
      tests_passed++;
    }

    // Test 6: Corrupted serialization - removed (no longer applicable)
    // Since ParseOmni now accepts AssimilationCtx directly, there's no way
    // to pass corrupted serialized data from the API level

    // Test 7: Valid case (control test - should succeed)
    tests_total++;
    wrp_cae::core::AssimilationCtx ctx7;
    ctx7.src = "file::" + kTestFileName;
    ctx7.dst = "iowarp::test_error_tag7";
    ctx7.format = "binary";
    ctx7.range_off = 0;
    ctx7.range_size = 0;
    if (TestErrorCase(cae_client, "ValidCase_Control", ctx7, false)) {
      tests_passed++;
    }

    // Cleanup
    HLOG(kInfo, "[CLEANUP] Removing test files...");
    std::remove(kTestFileName.c_str());
    if (fs::exists(kTestStoragePath)) {
      fs::remove(kTestStoragePath);
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception caught: {}", e.what());
    exit_code = 1;
  }

  // Print final results
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Tests passed: {}/{}", tests_passed, tests_total);

  if (tests_passed == tests_total && tests_total > 0) {
    HLOG(kSuccess, "TEST SUITE PASSED");
    exit_code = 0;
  } else {
    HLOG(kError, "TEST SUITE FAILED");
    exit_code = 1;
  }
  HLOG(kInfo, "========================================");

  return exit_code;
}
