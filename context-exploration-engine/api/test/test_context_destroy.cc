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
#include <hermes_shm/util/logging.h>

/**
 * Test that context_destroy can handle empty context list
 */
void test_empty_context_list() {
  HLOG(kInfo, "TEST: Empty context list");

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> empty_list;

  // Empty list should return success (0)
  int result = ctx_interface.ContextDestroy(empty_list);
  assert(result == 0 && "Empty context list should return success");

  HLOG(kSuccess, "PASSED: Empty context list test");
}

/**
 * Test that context_destroy handles non-existent contexts gracefully
 */
void test_nonexistent_context() {
  HLOG(kInfo, "TEST: Non-existent context");

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> contexts;
  contexts.push_back("definitely_does_not_exist_context_12345");

  // Non-existent context should be handled gracefully
  int result = ctx_interface.ContextDestroy(contexts);

  // Result could be 0 or non-zero depending on CTE behavior
  // Just verify the function completes without crashing
  HLOG(kInfo, "Destroy returned code: {}", result);
  HLOG(kSuccess, "PASSED: Non-existent context test");
}

/**
 * Test that context_destroy handles special characters
 */
void test_special_characters() {
  HLOG(kInfo, "TEST: Special characters");

  iowarp::ContextInterface ctx_interface;
  std::vector<std::string> contexts;
  contexts.push_back("test-context_with.special:chars");

  int result = ctx_interface.ContextDestroy(contexts);

  // Should handle special characters without crashing
  HLOG(kInfo, "Destroy returned code: {}", result);
  HLOG(kSuccess, "PASSED: Special characters test");
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  HLOG(kInfo, "========================================");
  HLOG(kInfo, "ContextInterface::ContextDestroy Tests");
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
    test_empty_context_list();

    test_nonexistent_context();

    test_special_characters();

    HLOG(kSuccess, "All tests PASSED!");
    return 0;
  } catch (const std::exception& e) {
    HLOG(kError, "Test FAILED with exception: {}", e.what());
    return 1;
  }
}
