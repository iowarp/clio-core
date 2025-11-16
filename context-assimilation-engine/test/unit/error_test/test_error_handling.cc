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
  std::cout << "\n--- Testing: " << test_name << " ---" << std::endl;

  // Call ParseOmni with vector containing single context
  std::vector<wrp_cae::core::AssimilationCtx> contexts = {ctx};
  chi::u32 num_tasks_scheduled = 0;
  chi::u32 result_code = cae_client.ParseOmni(HSHM_MCTX, contexts, num_tasks_scheduled);

  std::cout << "ParseOmni result: result_code=" << result_code
            << ", num_tasks=" << num_tasks_scheduled << std::endl;

  // Validate
  if (should_fail) {
    // We expect this to fail
    if (result_code != 0) {
      std::cout << "SUCCESS: " << test_name << " failed as expected (error code: "
                << result_code << ")" << std::endl;
      return true;
    } else {
      std::cerr << "ERROR: " << test_name << " should have failed but succeeded" << std::endl;
      return false;
    }
  } else {
    // We expect this to succeed
    if (result_code == 0) {
      std::cout << "SUCCESS: " << test_name << " succeeded as expected" << std::endl;
      return true;
    } else {
      std::cerr << "ERROR: " << test_name << " should have succeeded but failed (error code: "
                << result_code << ")" << std::endl;
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
  std::cout << "========================================" << std::endl;
  std::cout << "Error Handling ParseOmni Unit Test" << std::endl;
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

    // Generate test file (1MB)
    std::cout << "\n[SETUP] Generating test file..." << std::endl;
    const size_t file_size = 1024 * 1024;
    if (!GenerateTestFile(kTestFileName, file_size)) {
      std::cerr << "ERROR: Failed to generate test file" << std::endl;
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
        HSHM_MCTX,
        chi::PoolQuery::Local(),
        "test_cae_error_pool",
        wrp_cae::core::kCaePoolId,
        params);

    std::cout << "CAE pool created" << std::endl;

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
