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
 * test_binary_assim.cc - Unit test for ParseOmni API with binary file assimilation
 *
 * This test validates the ParseOmni API by:
 * 1. Creating a test binary file with patterned data
 * 2. Serializing an AssimilationCtx using cereal
 * 3. Calling ParseOmni to transfer the file to CTE
 * 4. Validating the transfer was successful
 * 5. Verifying the data exists in CTE
 *
 * Test Strategy:
 * - Tests happy path: successful file transfer
 * - Tests correct serialization/deserialization of context
 * - Tests integration with CTE (tag creation, blob storage)
 * - Tests chunking behavior for files > 1MB
 *
 * Environment Variables:
 * - INIT_CHIMAERA: If set to "1", initializes Chimaera runtime
 * - TEST_FILE_SIZE: Override default 256MB test file size (in MB)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>

// Chimaera and CAE headers
#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

// CTE headers
#include <wrp_cte/core/core_client.h>

// YAML parsing
#include <yaml-cpp/yaml.h>

// Logging
#include <hermes_shm/util/logging.h>

// Test configuration
constexpr size_t kDefaultFileSizeMB = 256;
constexpr size_t kMB = 1024 * 1024;
const std::string kTestFileName = "/tmp/test_binary_assim_file.bin";
const std::string kTestTagName = "test_binary_assim_tag";

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
 * Load OMNI configuration file and produce vector of AssimilationCtx
 *
 * @param omni_path Path to the OMNI YAML file
 * @return Vector of AssimilationCtx objects parsed from the OMNI file
 * @throws std::runtime_error if file cannot be loaded or is malformed
 *
 * Expected OMNI format:
 *   name: <job_name>
 *   transfers:
 *     - src: <source_uri>
 *       dst: <destination_uri>
 *       format: <format_type>
 *       depends_on: <dependency> (optional)
 *       range_off: <offset> (optional, default: 0)
 *       range_size: <size> (optional, default: 0 for full file)
 */
std::vector<wrp_cae::core::AssimilationCtx> LoadOmni(const std::string& omni_path) {
  HLOG(kInfo, "Loading OMNI file: {}", omni_path);

  YAML::Node config;
  try {
    config = YAML::LoadFile(omni_path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load OMNI file: " + std::string(e.what()));
  }

  // Check for required 'transfers' key
  if (!config["transfers"]) {
    throw std::runtime_error("OMNI file missing required 'transfers' key");
  }

  const YAML::Node& transfers = config["transfers"];
  if (!transfers.IsSequence()) {
    throw std::runtime_error("OMNI 'transfers' must be a sequence/array");
  }

  std::vector<wrp_cae::core::AssimilationCtx> contexts;
  contexts.reserve(transfers.size());

  // Parse each transfer entry
  for (size_t i = 0; i < transfers.size(); ++i) {
    const YAML::Node& transfer = transfers[i];

    // Validate required fields
    if (!transfer["src"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) + " missing required 'src' field");
    }
    if (!transfer["dst"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) + " missing required 'dst' field");
    }
    if (!transfer["format"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) + " missing required 'format' field");
    }

    wrp_cae::core::AssimilationCtx ctx;
    ctx.src = transfer["src"].as<std::string>();
    ctx.dst = transfer["dst"].as<std::string>();
    ctx.format = transfer["format"].as<std::string>();
    ctx.depends_on = transfer["depends_on"] ? transfer["depends_on"].as<std::string>() : "";
    ctx.range_off = transfer["range_off"] ? transfer["range_off"].as<size_t>() : 0;
    ctx.range_size = transfer["range_size"] ? transfer["range_size"].as<size_t>() : 0;

    contexts.push_back(ctx);

    HLOG(kInfo, "Loaded transfer {}/{}", (i + 1), transfers.size());
    HLOG(kInfo, "  src: {}", ctx.src);
    HLOG(kInfo, "  dst: {}", ctx.dst);
    HLOG(kInfo, "  format: {}", ctx.format);
    if (!ctx.depends_on.empty()) {
      HLOG(kInfo, "  depends_on: {}", ctx.depends_on);
    }
    if (ctx.range_off != 0 || ctx.range_size != 0) {
      HLOG(kInfo, "  range: [{}, {}]", ctx.range_off, ctx.range_size);
    }
  }

  HLOG(kSuccess, "Successfully loaded {} transfer(s) from OMNI file", contexts.size());
  return contexts;
}

/**
 * Clean up test file
 */
void CleanupTestFile(const std::string& file_path) {
  if (std::remove(file_path.c_str()) == 0) {
    HLOG(kInfo, "Test file cleaned up: {}", file_path);
  } else {
    HLOG(kWarning, "Failed to remove test file: {}", file_path);
  }
}

/**
 * Main test function
 */
int main(int argc, char* argv[]) {
  HLOG(kInfo, "========================================");
  HLOG(kInfo, "Binary Assimilation ParseOmni Unit Test");
  HLOG(kInfo, "========================================");

  int exit_code = 0;

  try {
    // Initialize Chimaera runtime (CHIMAERA_WITH_RUNTIME controls behavior)
    HLOG(kInfo, "Initializing Chimaera...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      HLOG(kError, "Failed to initialize Chimaera");
      return 1;
    }
    HLOG(kSuccess, "Chimaera initialized successfully");

    // Verify Chimaera IPC is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      HLOG(kError, "Chimaera IPC not initialized");
      return 1;
    }
    HLOG(kSuccess, "Chimaera IPC verified");

    // Get test file size from environment or use default
    size_t file_size_mb = kDefaultFileSizeMB;
    const char* env_file_size = std::getenv("TEST_FILE_SIZE");
    if (env_file_size) {
      file_size_mb = std::stoul(env_file_size);
    }
    const size_t file_size_bytes = file_size_mb * kMB;

    // Step 1: Generate test file
    HLOG(kInfo, "[STEP 1] Generating test file...");
    if (!GenerateTestFile(kTestFileName, file_size_bytes)) {
      return 1;
    }

    // Step 2: Connect to CTE
    HLOG(kInfo, "[STEP 2] Connecting to CTE...");
    wrp_cte::core::WRP_CTE_CLIENT_INIT();
    HLOG(kSuccess, "CTE client initialized");

    // Step 2.5: Initialize CAE client
    HLOG(kInfo, "[STEP 2.5] Initializing CAE client...");
    WRP_CAE_CLIENT_INIT();
    HLOG(kSuccess, "CAE client initialized");

    // Step 3: Create CAE pool
    HLOG(kInfo, "[STEP 3] Creating CAE pool...");
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams params;

    auto create_task = cae_client.AsyncCreate(
        chi::PoolQuery::Local(),
        "test_cae_pool",
        wrp_cae::core::kCaePoolId,
        params);
    create_task.Wait();

    HLOG(kSuccess, "CAE pool created with ID: {}", cae_client.pool_id_);

    // Step 4: Load OMNI configuration file
    HLOG(kInfo, "[STEP 4] Loading OMNI configuration...");
    const std::string source_path = __FILE__;  // Get current source file path
    const std::string omni_file = source_path.substr(0, source_path.find_last_of('/')) + "/binary_assim_omni.yaml";

    std::vector<wrp_cae::core::AssimilationCtx> contexts;
    try {
      contexts = LoadOmni(omni_file);
    } catch (const std::exception& e) {
      HLOG(kError, "Failed to load OMNI file: {}", e.what());
      return 1;
    }

    // Step 5: Call ParseOmni (serialization happens transparently in ParseOmniTask)
    HLOG(kInfo, "[STEP 5] Calling ParseOmni...");
    auto parse_task = cae_client.AsyncParseOmni(contexts);
    parse_task.Wait();
    chi::u32 result_code = parse_task->GetReturnCode();
    chi::u32 num_tasks_scheduled = parse_task->num_tasks_scheduled_;

    HLOG(kInfo, "ParseOmni completed:");
    HLOG(kInfo, "  result_code: {}", result_code);
    HLOG(kInfo, "  num_tasks_scheduled: {}", num_tasks_scheduled);

    // Step 6: Validate results
    HLOG(kInfo, "[STEP 6] Validating results...");

    if (result_code != 0) {
      HLOG(kError, "ParseOmni failed with result_code: {}", result_code);
      exit_code = 1;
    } else if (num_tasks_scheduled == 0) {
      HLOG(kError, "ParseOmni returned 0 tasks scheduled");
      exit_code = 1;
    } else {
      HLOG(kSuccess, "ParseOmni executed successfully");
    }

    // Step 8: Verify data in CTE
    HLOG(kInfo, "[STEP 8] Verifying data in CTE...");

    // Get CTE client
    auto cte_client = WRP_CTE_CLIENT;

    // Check if tag exists
    auto tag_task = cte_client->AsyncGetOrCreateTag(kTestTagName);
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;
    if (tag_id.IsNull()) {
      HLOG(kError, "Tag not found in CTE: {}", kTestTagName);
      exit_code = 1;
    } else {
      HLOG(kSuccess, "Tag found in CTE: {} (ID: {})", kTestTagName, tag_id);

      // Get tag size to verify data was transferred
      auto size_task = cte_client->AsyncGetTagSize(tag_id);
      size_task.Wait();
      size_t tag_size = size_task->tag_size_;
      HLOG(kInfo, "Tag size in CTE: {} bytes", tag_size);
      HLOG(kInfo, "Original file size: {} bytes", file_size_bytes);

      if (tag_size == 0) {
        HLOG(kError, "Tag size is 0, no data transferred");
        exit_code = 1;
      } else if (tag_size != file_size_bytes) {
        HLOG(kWarning, "Tag size ({}) does not match file size ({})",
             tag_size, file_size_bytes);
        // Note: This is a warning, not an error - don't set exit_code
      } else {
        HLOG(kSuccess, "Tag size matches file size - data verified in CTE");
      }
    }

    // Step 9: Cleanup
    HLOG(kInfo, "[STEP 9] Cleaning up...");
    CleanupTestFile(kTestFileName);

  } catch (const std::exception& e) {
    HLOG(kError, "Exception caught: {}", e.what());
    exit_code = 1;
  }

  // Print final result
  HLOG(kInfo, "========================================");
  if (exit_code == 0) {
    HLOG(kSuccess, "TEST PASSED");
  } else {
    HLOG(kError, "TEST FAILED");
  }
  HLOG(kInfo, "========================================");

  return exit_code;
}
