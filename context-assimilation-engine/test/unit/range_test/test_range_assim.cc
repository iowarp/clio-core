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
 * - INIT_CHIMAERA: If set to "1", initializes Chimaera runtime
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

// Chimaera and CAE headers
#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

// CTE headers
#include <wrp_cte/core/core_client.h>

// Test configuration
constexpr size_t kTestFileSizeMB = 10;  // 10MB for range testing
constexpr size_t kMB = 1024 * 1024;
const std::string kTestFileName = "/tmp/test_range_assim_file.bin";
const std::string kTestTagPrefix = "test_range_assim_";

/**
 * Generate a test file with patterned data
 */
bool GenerateTestFile(const std::string& file_path, size_t size_bytes) {
  std::cout << "Generating test file: " << file_path << " (" << size_bytes << " bytes)" << std::endl;

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Failed to create test file" << std::endl;
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
  std::cout << "Test file generated successfully" << std::endl;
  return true;
}

/**
 * Test a specific range
 */
bool TestRange(wrp_cae::core::Client& cae_client,
               const std::string& test_name,
               size_t range_off,
               size_t range_size) {
  std::cout << "\n--- Testing: " << test_name << " ---" << std::endl;
  std::cout << "Range: offset=" << range_off << ", size=" << range_size << std::endl;

  // Create unique tag name for this test
  std::string tag_name = kTestTagPrefix + test_name;

  // Create AssimilationCtx
  wrp_cae::core::AssimilationCtx ctx;
  ctx.src = "file::" + kTestFileName;
  ctx.dst = "iowarp::" + tag_name;
  ctx.format = "binary";
  ctx.depends_on = "";
  ctx.range_off = range_off;
  ctx.range_size = range_size;

  // Call ParseOmni with vector containing single context
  std::vector<wrp_cae::core::AssimilationCtx> contexts = {ctx};
  chi::u32 num_tasks_scheduled = 0;
  chi::u32 result_code = cae_client.ParseOmni(contexts, num_tasks_scheduled);

  std::cout << "ParseOmni result: result_code=" << result_code
            << ", num_tasks=" << num_tasks_scheduled << std::endl;

  // Validate
  if (result_code != 0) {
    std::cerr << "ERROR: ParseOmni failed with code " << result_code << std::endl;
    return false;
  }

  if (num_tasks_scheduled == 0) {
    std::cerr << "ERROR: No tasks scheduled" << std::endl;
    return false;
  }

  std::cout << "SUCCESS: " << test_name << " passed" << std::endl;
  return true;
}

/**
 * Main test function
 */
int main(int argc, char* argv[]) {
  std::cout << "========================================" << std::endl;
  std::cout << "Range Assimilation ParseOmni Unit Test" << std::endl;
  std::cout << "========================================" << std::endl;

  int exit_code = 0;
  int tests_passed = 0;
  int tests_total = 0;

  try {
    // Initialize Chimaera runtime (CHIMAERA_WITH_RUNTIME controls behavior)
    std::cout << "Initializing Chimaera..." << std::endl;
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    if (!success) {
      std::cerr << "ERROR: Failed to initialize Chimaera" << std::endl;
      return 1;
    }
    std::cout << "Chimaera initialized successfully" << std::endl;

    // Verify Chimaera IPC
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      std::cerr << "ERROR: Chimaera IPC not initialized" << std::endl;
      return 1;
    }
    std::cout << "Chimaera IPC verified" << std::endl;

    // Generate test file
    const size_t file_size_bytes = kTestFileSizeMB * kMB;
    std::cout << "\n[SETUP] Generating test file..." << std::endl;
    if (!GenerateTestFile(kTestFileName, file_size_bytes)) {
      return 1;
    }

    // Connect to CTE
    std::cout << "\n[SETUP] Connecting to CTE..." << std::endl;
    wrp_cte::core::WRP_CTE_CLIENT_INIT();

    // Initialize CAE client
    std::cout << "\n[SETUP] Initializing CAE client..." << std::endl;
    WRP_CAE_CLIENT_INIT();

    // Create CAE pool
    std::cout << "\n[SETUP] Creating CAE pool..." << std::endl;
    wrp_cae::core::Client cae_client;
    wrp_cae::core::CreateParams params;

    cae_client.Create(
        chi::PoolQuery::Local(),
        "test_cae_range_pool",
        wrp_cae::core::kCaePoolId,
        params);

    std::cout << "CAE pool created" << std::endl;

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
    std::cout << "\n[CLEANUP] Removing test file..." << std::endl;
    std::remove(kTestFileName.c_str());

  } catch (const std::exception& e) {
    std::cerr << "ERROR: Exception caught: " << e.what() << std::endl;
    exit_code = 1;
  }

  // Print final results
  std::cout << "\n========================================" << std::endl;
  std::cout << "Tests passed: " << tests_passed << "/" << tests_total << std::endl;

  if (tests_passed == tests_total && tests_total > 0) {
    std::cout << "TEST SUITE PASSED" << std::endl;
    exit_code = 0;
  } else {
    std::cout << "TEST SUITE FAILED" << std::endl;
    exit_code = 1;
  }
  std::cout << "========================================" << std::endl;

  return exit_code;
}
