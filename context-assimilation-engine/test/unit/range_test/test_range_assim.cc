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
 * test_range_assim.cc - Unit test for ParseOmni API with range-based assimilation
 *
 * This test validates the ParseOmni API with partial file transfers using
 * range_off and range_size parameters.
 *
 * Test Strategy:
 * - Tests partial file transfer (middle chunk)
 * - Tests range boundaries (first byte, last byte)
 * - Tests range validation
 * - Tests offset + size combinations
 *
 * Environment Variables:
 * - INIT_CLIO: If set to "1", initializes CLIO Runtime runtime
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

// CLIO Runtime and CAE headers
#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>

// CTE headers
#include <clio_cte/core/core_client.h>

// Logging
#include <clio_ctp/util/logging.h>

#include <filesystem>

// Test configuration
constexpr size_t kTestFileSizeMB = 10;  // 10MB for range testing
constexpr size_t kMB = 1024 * 1024;
const std::string kTestFileName =
    (std::filesystem::temp_directory_path() / "test_range_assim_file.bin")
        .string();
const std::string kTestTagPrefix = "test_range_assim_";

/**
 * Generate a test file with patterned data
 */
bool GenerateTestFile(const std::string& file_path, size_t size_bytes) {
  HLOG(kInfo, "Generating test file: {} ({} bytes)", file_path, size_bytes);

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    HLOG(kError, "Failed to create test file");
    return false;
  }

  // Generate patterned data (4-byte blocks with incrementing indices)
  const size_t block_size = 4;
  const size_t num_blocks = size_bytes / block_size;

  for (size_t i = 0; i < num_blocks; ++i) {
    uint32_t value = static_cast<uint32_t>(i);
    file.write(reinterpret_cast<const char*>(&value), block_size);
  }

  file.close();
  HLOG(kSuccess, "Test file generated successfully");
  return true;
}

/**
 * Test a specific range
 */
bool TestRange(clio::cae::core::Client& cae_client,
               const std::string& test_name,
               size_t range_off,
               size_t range_size) {
  HLOG(kInfo, "--- Testing: {} ---", test_name);
  HLOG(kInfo, "Range: offset={}, size={}", range_off, range_size);

  // Create unique tag name for this test
  std::string tag_name = kTestTagPrefix + test_name;

  // Create AssimilationCtx
  clio::cae::core::AssimilationCtx ctx;
  ctx.src = "file::" + kTestFileName;
  ctx.dst = "iowarp::" + tag_name;
  ctx.format = "binary";
  ctx.depends_on = "";
  ctx.range_off = range_off;
  ctx.range_size = range_size;

  // Call ParseOmni with vector containing single context
  std::vector<clio::cae::core::AssimilationCtx> contexts = {ctx};
  auto parse_task = cae_client.AsyncParseOmni(contexts);
  parse_task.Wait();
  clio::run::u32 result_code = parse_task->GetReturnCode();
  clio::run::u32 num_tasks_scheduled = parse_task->num_tasks_scheduled_;

  HLOG(kInfo, "ParseOmni result: result_code={}, num_tasks={}", result_code, num_tasks_scheduled);

  // Validate
  if (result_code != 0) {
    HLOG(kError, "ParseOmni failed with code {}", result_code);
    return false;
  }

  if (num_tasks_scheduled == 0) {
    HLOG(kError, "No tasks scheduled");
    return false;
  }

  HLOG(kSuccess, "{} passed", test_name);
  return true;
}

/**
 * Main test function
 */
int main(int argc, char* argv[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Range Assimilation ParseOmni Unit Test");
  HLOG(kInfo, "========================================");

  int exit_code = 0;
  int tests_passed = 0;
  int tests_total = 0;

  try {
    // Initialize CLIO Runtime runtime (CLIO_WITH_RUNTIME controls behavior)
    HLOG(kInfo, "Initializing Clio...");
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    if (!success) {
      HLOG(kError, "Failed to initialize Clio");
      return 1;
    }
    HLOG(kSuccess, "Clio initialized successfully");

    // Verify CLIO Runtime IPC
    auto* ipc_manager = CLIO_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Clio IPC not initialized");
      return 1;
    }
    HLOG(kSuccess, "Clio IPC verified");

    // Generate test file
    const size_t file_size_bytes = kTestFileSizeMB * kMB;
    HLOG(kInfo, "[SETUP] Generating test file...");
    if (!GenerateTestFile(kTestFileName, file_size_bytes)) {
      return 1;
    }

    // Connect to CTE
    HLOG(kInfo, "[SETUP] Connecting to CTE...");
    clio::cte::core::CLIO_CTE_CLIENT_INIT();

    // Initialize CAE client
    HLOG(kInfo, "[SETUP] Initializing CAE client...");
    CLIO_CAE_CLIENT_INIT();

    // Create CAE pool
    HLOG(kInfo, "[SETUP] Creating CAE pool...");
    clio::cae::core::Client cae_client;
    clio::cae::core::CreateParams params;

    auto create_task = cae_client.AsyncCreate(
        clio::run::PoolQuery::Local(),
        "test_cae_range_pool",
        clio::cae::core::kCaePoolId,
        params);
    create_task.Wait();

    HLOG(kSuccess, "CAE pool created");

    // Test 1: Middle chunk (1MB from offset 2MB)
    tests_total++;
    if (TestRange(cae_client, "middle_chunk", 2 * kMB, 1 * kMB)) {
      tests_passed++;
    }

    // Test 2: First byte
    tests_total++;
    if (TestRange(cae_client, "first_byte", 0, 1)) {
      tests_passed++;
    }

    // Test 3: Last 1KB
    tests_total++;
    if (TestRange(cae_client, "last_1kb", file_size_bytes - 1024, 1024)) {
      tests_passed++;
    }

    // Test 4: First 512KB
    tests_total++;
    if (TestRange(cae_client, "first_512kb", 0, 512 * 1024)) {
      tests_passed++;
    }

    // Test 5: Offset at 1MB boundary, size 2MB
    tests_total++;
    if (TestRange(cae_client, "aligned_2mb", 1 * kMB, 2 * kMB)) {
      tests_passed++;
    }

    // Cleanup
    HLOG(kInfo, "[CLEANUP] Removing test file...");
    std::remove(kTestFileName.c_str());

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

  // No-op on POSIX; on Windows skips the libzmq teardown abort.
  ctp::SystemInfo::TerminateProcessNow(exit_code);
  return exit_code;
}
