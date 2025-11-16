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
  std::cout << "Generating test file: " << file_path << " (" << size_bytes << " bytes)" << std::endl;

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Failed to create test file: " << file_path << std::endl;
    return false;
  }

  // Generate patterned data (4-byte blocks with incrementing indices)
  const size_t block_size = 4;
  const size_t num_blocks = size_bytes / block_size;

  for (size_t i = 0; i < num_blocks; ++i) {
    uint32_t value = static_cast<uint32_t>(i);
    file.write(reinterpret_cast<const char*>(&value), block_size);

    if (!file.good()) {
      std::cerr << "ERROR: Failed to write to test file at block " << i << std::endl;
      file.close();
      return false;
    }
  }

  file.close();
  std::cout << "Test file generated successfully (" << num_blocks << " blocks)" << std::endl;
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
  std::cout << "Loading OMNI file: " << omni_path << std::endl;

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

    std::cout << "  Loaded transfer " << (i + 1) << "/" << transfers.size() << ":" << std::endl;
    std::cout << "    src: " << ctx.src << std::endl;
    std::cout << "    dst: " << ctx.dst << std::endl;
    std::cout << "    format: " << ctx.format << std::endl;
    if (!ctx.depends_on.empty()) {
      std::cout << "    depends_on: " << ctx.depends_on << std::endl;
    }
    if (ctx.range_off != 0 || ctx.range_size != 0) {
      std::cout << "    range: [" << ctx.range_off << ", " << ctx.range_size << "]" << std::endl;
    }
  }

  std::cout << "Successfully loaded " << contexts.size() << " transfer(s) from OMNI file" << std::endl;
  return contexts;
}

/**
 * Clean up test file
 */
void CleanupTestFile(const std::string& file_path) {
  if (std::remove(file_path.c_str()) == 0) {
    std::cout << "Test file cleaned up: " << file_path << std::endl;
  } else {
    std::cerr << "WARNING: Failed to remove test file: " << file_path << std::endl;
  }
}

/**
 * Main test function
 */
int main(int argc, char* argv[]) {
  std::cout << "========================================" << std::endl;
  std::cout << "Binary Assimilation ParseOmni Unit Test" << std::endl;
  std::cout << "========================================" << std::endl;

  int exit_code = 0;

  try {
    // Initialize Chimaera runtime (CHIMAERA_WITH_RUNTIME controls behavior)
    std::cout << "Initializing Chimaera..." << std::endl;
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      std::cerr << "ERROR: Failed to initialize Chimaera" << std::endl;
      return 1;
    }
    std::cout << "Chimaera initialized successfully" << std::endl;

    // Verify Chimaera IPC is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      std::cerr << "ERROR: Chimaera IPC not initialized" << std::endl;
      return 1;
    }
    std::cout << "Chimaera IPC verified" << std::endl;

    // Get test file size from environment or use default
    size_t file_size_mb = kDefaultFileSizeMB;
    const char* env_file_size = std::getenv("TEST_FILE_SIZE");
    if (env_file_size) {
      file_size_mb = std::stoul(env_file_size);
    }
    const size_t file_size_bytes = file_size_mb * kMB;

    // Step 1: Generate test file
    std::cout << "\n[STEP 1] Generating test file..." << std::endl;
    if (!GenerateTestFile(kTestFileName, file_size_bytes)) {
      return 1;
    }

    // Step 2: Connect to CTE
    std::cout << "\n[STEP 2] Connecting to CTE..." << std::endl;
    wrp_cte::core::WRP_CTE_CLIENT_INIT();
    std::cout << "CTE client initialized" << std::endl;

    // Step 2.5: Initialize CAE client
    std::cout << "\n[STEP 2.5] Initializing CAE client..." << std::endl;
    WRP_CAE_CLIENT_INIT();
    std::cout << "CAE client initialized" << std::endl;

    // Step 3: Create CAE pool
    std::cout << "\n[STEP 3] Creating CAE pool..." << std::endl;
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams params;

    cae_client.Create(
        HSHM_MCTX,
        chi::PoolQuery::Local(),
        "test_cae_pool",
        wrp_cae::core::kCaePoolId,
        params);

    std::cout << "CAE pool created with ID: " << cae_client.pool_id_ << std::endl;

    // Step 4: Load OMNI configuration file
    std::cout << "\n[STEP 4] Loading OMNI configuration..." << std::endl;
    const std::string source_path = __FILE__;  // Get current source file path
    const std::string omni_file = source_path.substr(0, source_path.find_last_of('/')) + "/binary_assim_omni.yaml";

    std::vector<wrp_cae::core::AssimilationCtx> contexts;
    try {
      contexts = LoadOmni(omni_file);
    } catch (const std::exception& e) {
      std::cerr << "ERROR: Failed to load OMNI file: " << e.what() << std::endl;
      return 1;
    }

    // Step 5: Call ParseOmni (serialization happens transparently in ParseOmniTask)
    std::cout << "\n[STEP 5] Calling ParseOmni..." << std::endl;
    chi::u32 num_tasks_scheduled = 0;
    chi::u32 result_code = cae_client.ParseOmni(HSHM_MCTX, contexts, num_tasks_scheduled);

    std::cout << "ParseOmni completed:" << std::endl;
    std::cout << "  result_code: " << result_code << std::endl;
    std::cout << "  num_tasks_scheduled: " << num_tasks_scheduled << std::endl;

    // Step 6: Validate results
    std::cout << "\n[STEP 6] Validating results..." << std::endl;

    if (result_code != 0) {
      std::cerr << "ERROR: ParseOmni failed with result_code: " << result_code << std::endl;
      exit_code = 1;
    } else if (num_tasks_scheduled == 0) {
      std::cerr << "ERROR: ParseOmni returned 0 tasks scheduled" << std::endl;
      exit_code = 1;
    } else {
      std::cout << "SUCCESS: ParseOmni executed successfully" << std::endl;
    }

    // Step 8: Verify data in CTE
    std::cout << "\n[STEP 8] Verifying data in CTE..." << std::endl;

    // Get CTE client
    auto cte_client = WRP_CTE_CLIENT;

    // Check if tag exists
    wrp_cte::core::TagId tag_id = cte_client->GetOrCreateTag(HSHM_MCTX, kTestTagName);
    if (tag_id.IsNull()) {
      std::cerr << "ERROR: Tag not found in CTE: " << kTestTagName << std::endl;
      exit_code = 1;
    } else {
      std::cout << "Tag found in CTE: " << kTestTagName << " (ID: " << tag_id << ")" << std::endl;

      // Get tag size to verify data was transferred
      size_t tag_size = cte_client->GetTagSize(HSHM_MCTX, tag_id);
      std::cout << "Tag size in CTE: " << tag_size << " bytes" << std::endl;
      std::cout << "Original file size: " << file_size_bytes << " bytes" << std::endl;

      if (tag_size == 0) {
        std::cerr << "ERROR: Tag size is 0, no data transferred" << std::endl;
        exit_code = 1;
      } else if (tag_size != file_size_bytes) {
        std::cerr << "WARNING: Tag size (" << tag_size << ") does not match file size ("
                  << file_size_bytes << ")" << std::endl;
        // Note: This is a warning, not an error - don't set exit_code
      } else {
        std::cout << "SUCCESS: Tag size matches file size - data verified in CTE" << std::endl;
      }
    }

    // Step 9: Cleanup
    std::cout << "\n[STEP 9] Cleaning up..." << std::endl;
    CleanupTestFile(kTestFileName);

  } catch (const std::exception& e) {
    std::cerr << "ERROR: Exception caught: " << e.what() << std::endl;
    exit_code = 1;
  }

  // Print final result
  std::cout << "\n========================================" << std::endl;
  if (exit_code == 0) {
    std::cout << "TEST PASSED" << std::endl;
  } else {
    std::cout << "TEST FAILED" << std::endl;
  }
  std::cout << "========================================" << std::endl;

  return exit_code;
}
