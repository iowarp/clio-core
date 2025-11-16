/**
 * test_context_destroy.cc - Unit test for ContextInterface destroy API
 *
 * This test validates the ContextDestroy API by:
 * 1. Testing empty context list handling
 * 2. Testing non-existent context handling
 * 3. Testing special characters in context names
 *
 * Environment Variables:
 * - INIT_CHIMAERA: If set to "1", initializes Chimaera runtime
 */

#include <wrp_cee/api/context_interface.h>
#include <chimaera/chimaera.h>
#include <iostream>
#include <cassert>
#include <cstring>

/**
 * Test that context_destroy can handle empty context list
 */
void test_empty_context_list() {
  std::cout << "TEST: Empty context list" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> empty_list;

  // Empty list should return success (0)
  int result = ctx_interface.ContextDestroy(empty_list);
  assert(result == 0 && "Empty context list should return success");

  std::cout << "  PASSED: Empty context list test" << std::endl;
}

/**
 * Test that context_destroy handles non-existent contexts gracefully
 */
void test_nonexistent_context() {
  std::cout << "TEST: Non-existent context" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> contexts;
  contexts.push_back("definitely_does_not_exist_context_12345");

  // Non-existent context should be handled gracefully
  int result = ctx_interface.ContextDestroy(contexts);

  // Result could be 0 or non-zero depending on CTE behavior
  // Just verify the function completes without crashing
  std::cout << "  Destroy returned code: " << result << std::endl;
  std::cout << "  PASSED: Non-existent context test" << std::endl;
}

/**
 * Test that context_destroy handles special characters
 */
void test_special_characters() {
  std::cout << "TEST: Special characters" << std::endl;

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> contexts;
  contexts.push_back("test-context_with.special:chars");

  int result = ctx_interface.ContextDestroy(contexts);

  // Should handle special characters without crashing
  std::cout << "  Destroy returned code: " << result << std::endl;
  std::cout << "  PASSED: Special characters test" << std::endl;
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  std::cout << "========================================" << std::endl;
  std::cout << "ContextInterface::ContextDestroy Tests" << std::endl;
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
    test_empty_context_list();
    std::cout << std::endl;

    test_nonexistent_context();
    std::cout << std::endl;

    test_special_characters();
    std::cout << std::endl;

    std::cout << "All tests PASSED!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
    return 1;
  }
}
