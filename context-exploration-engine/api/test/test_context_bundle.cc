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

// Test configuration
const std::string kTestFileName = "/tmp/test_cee_bundle_file.bin";
const std::string kTestTagName = "cee_test_bundle_tag";
const size_t kTestFileSize = 1024 * 1024;  // 1 MB test file

/**
 * Generate a test file with patterned data for validation
 * Pattern: Each 4-byte block contains the block index (little endian)
 */
bool GenerateTestFile(const std::string& file_path, size_t size_bytes) {
  std::cout << "  Generating test file: " << file_path << " (" << size_bytes << " bytes)" << std::endl;

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "  ERROR: Failed to create test file: " << file_path << std::endl;
    return false;
  }

  // Generate patterned data (4-byte blocks with incrementing indices)
  const size_t block_size = 4;
  const size_t num_blocks = size_bytes / block_size;

  for (size_t i = 0; i < num_blocks; ++i) {
    uint32_t value = static_cast<uint32_t>(i);
    file.write(reinterpret_cast<const char*>(&value), block_size);

    if (!file.good()) {
      std::cerr << "  ERROR: Failed to write to test file at block " << i << std::endl;
      file.close();
      return false;
    }
  }

  file.close();
  std::cout << "  Test file generated successfully (" << num_blocks << " blocks)" << std::endl;
  return true;
}

/**
 * Test that context_bundle can handle empty bundles
 */
void test_empty_bundle() {
  std::cout << "TEST: Empty bundle" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<wrp_cae::core::AssimilationCtx> empty_bundle;

  // Empty bundle should return success (0)
  int result = ctx_interface.ContextBundle(empty_bundle);
  assert(result == 0 && "Empty bundle should return success");

  std::cout << "  PASSED: Empty bundle test" << std::endl;
}

/**
 * Test AssimilationCtx constructor with all parameters
 */
void test_assimilation_ctx_constructor() {
  std::cout << "TEST: AssimilationCtx constructor" << std::endl;

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

  std::cout << "  PASSED: AssimilationCtx constructor test" << std::endl;
}

/**
 * Test comprehensive bundle-and-retrieve workflow
 * This is the main integration test that validates:
 * 1. File creation
 * 2. Bundle operation (assimilation)
 * 3. Query operation (verification)
 */
void test_bundle_and_retrieve_workflow() {
  std::cout << "TEST: Bundle-and-retrieve workflow" << std::endl;

  // Step 1: Generate test file
  std::cout << "  [STEP 1] Generating test file..." << std::endl;
  if (!GenerateTestFile(kTestFileName, kTestFileSize)) {
    assert(false && "Failed to generate test file");
  }

  // Step 2: Initialize CTE client
  std::cout << "  [STEP 2] Initializing CTE client..." << std::endl;
  wrp_cte::core::WRP_CTE_CLIENT_INIT();

  // Step 2.5: Register a RAM storage target with CTE
  std::cout << "  [STEP 2.5] Registering RAM storage target with CTE..." << std::endl;
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
  std::cout << "  Storage target registered successfully" << std::endl;

  // Step 3: Create CAE pool
  std::cout << "  [STEP 3] Creating CAE pool..." << std::endl;
  wrp_cae::core::Client cae_client;
  wrp_cae::core::CreateParams params;

  auto create_task = cae_client.AsyncCreate(
      chi::PoolQuery::Local(),
      "test_cee_cae_pool",
      wrp_cae::core::kCaePoolId,
      params);
  create_task.Wait();

  std::cout << "  CAE pool created with ID: " << cae_client.pool_id_ << std::endl;

  // Step 4: Bundle the test file using ContextInterface
  std::cout << "  [STEP 4] Bundling test file..." << std::endl;
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
  std::cout << "  Bundle operation completed successfully" << std::endl;

  // Step 5: Wait for assimilation to complete
  std::cout << "  [STEP 5] Waiting for assimilation to complete..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Step 6: Query to verify the data was ingested
  std::cout << "  [STEP 6] Querying for bundled data..." << std::endl;
  std::vector<std::string> query_results = ctx_interface.ContextQuery(
      kTestTagName,  // Exact tag name
      ".*");         // Match all blobs

  std::cout << "  Query returned " << query_results.size() << " results" << std::endl;

  // Verify we got at least one result
  if (query_results.size() > 0) {
    std::cout << "  Found blobs in tag '" << kTestTagName << "':" << std::endl;
    for (const auto& blob : query_results) {
      std::cout << "    - " << blob << std::endl;
    }
  }

  // Note: The number of blobs depends on chunking behavior
  // For a 1MB file, it might be stored as multiple blobs
  assert(query_results.size() > 0 && "Query should find at least one blob after bundling");

  // Step 7: Cleanup - destroy the context
  std::cout << "  [STEP 7] Cleaning up test context..." << std::endl;
  std::vector<std::string> contexts_to_delete = {kTestTagName};
  int destroy_result = ctx_interface.ContextDestroy(contexts_to_delete);
  std::cout << "  Destroy returned code: " << destroy_result << std::endl;

  // Step 8: Delete test file
  std::remove(kTestFileName.c_str());

  std::cout << "  PASSED: Bundle-and-retrieve workflow test" << std::endl;
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  std::cout << "========================================" << std::endl;
  std::cout << "ContextInterface::ContextBundle Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  try {
    // Initialize Chimaera runtime if requested (for unit tests)
    const char* init_chimaera = std::getenv("INIT_CHIMAERA");
    if (init_chimaera && std::strcmp(init_chimaera, "1") == 0) {
      std::cout << "Initializing Chimaera (INIT_CHIMAERA=1)..." << std::endl;
      chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      std::cout << "Chimaera initialized" << std::endl;
    }

    // Verify Chimaera IPC is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      std::cerr << "ERROR: Chimaera IPC not initialized. Is the runtime running?" << std::endl;
      std::cerr << "HINT: Set INIT_CHIMAERA=1 to initialize runtime or start runtime externally" << std::endl;
      return 1;
    }
    std::cout << "Chimaera IPC verified\n" << std::endl;

    // Run all tests
    test_empty_bundle();
    std::cout << std::endl;

    test_assimilation_ctx_constructor();
    std::cout << std::endl;

    test_bundle_and_retrieve_workflow();
    std::cout << std::endl;

    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
    return 1;
  }
}
