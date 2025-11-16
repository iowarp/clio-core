/**
 * test_context_query.cc - Unit test for ContextInterface query API
 *
 * This test validates the ContextQuery API by:
 * 1. Calling ContextQuery with various patterns
 * 2. Verifying the function completes without crashes
 * 3. Testing different regex patterns
 *
 * Environment Variables:
 * - INIT_CHIMAERA: If set to "1", initializes Chimaera runtime
 */

#include <wrp_cee/api/context_interface.h>
#include <chimaera/chimaera.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <set>

/**
 * Test that context_query can be called and returns a vector
 */
void test_basic_query() {
  std::cout << "TEST: Basic query" << std::endl;

  iowarp::ContextInterface ctx_interface;

  // Query for all tags and blobs using wildcard patterns
  std::vector<std::string> results = ctx_interface.ContextQuery(".*", ".*");

  // Result should be a vector (may be empty if no tags exist)
  // Just verify the function doesn't crash
  std::cout << "  Query returned " << results.size() << " results" << std::endl;
  std::cout << "  PASSED: Basic query test" << std::endl;
}

/**
 * Test that context_query handles specific patterns
 */
void test_specific_patterns() {
  std::cout << "TEST: Specific patterns" << std::endl;

  iowarp::ContextInterface ctx_interface;

  // Query for specific patterns
  std::vector<std::string> results1 = ctx_interface.ContextQuery("test_.*", ".*");
  std::vector<std::string> results2 = ctx_interface.ContextQuery(".*", "blob_[0-9]+");
  std::vector<std::string> results3 = ctx_interface.ContextQuery("my_tag", "my_blob");

  // Just verify the function completes without crashing
  std::cout << "  Pattern 1 returned " << results1.size() << " results" << std::endl;
  std::cout << "  Pattern 2 returned " << results2.size() << " results" << std::endl;
  std::cout << "  Pattern 3 returned " << results3.size() << " results" << std::endl;
  std::cout << "  PASSED: Specific patterns test" << std::endl;
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  std::cout << "========================================" << std::endl;
  std::cout << "ContextInterface::ContextQuery Tests" << std::endl;
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
    test_basic_query();
    std::cout << std::endl;

    test_specific_patterns();
    std::cout << std::endl;

    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
    return 1;
  }
}
