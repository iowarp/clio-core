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
#include <hermes_shm/util/logging.h>

/**
 * Test that context_query can be called and returns a vector
 */
void test_basic_query() {
  HLOG(kInfo, "TEST: Basic query");

  iowarp::ContextInterface ctx_interface;

  // Query for all tags and blobs using wildcard patterns
  std::vector<std::string> results = ctx_interface.ContextQuery(".*", ".*");

  // Result should be a vector (may be empty if no tags exist)
  // Just verify the function doesn't crash
  HLOG(kInfo, "Query returned {} results", results.size());
  HLOG(kSuccess, "PASSED: Basic query test");
}

/**
 * Test that context_query handles specific patterns
 */
void test_specific_patterns() {
  HLOG(kInfo, "TEST: Specific patterns");

  iowarp::ContextInterface ctx_interface;

  // Query for specific patterns
  std::vector<std::string> results1 = ctx_interface.ContextQuery("test_.*", ".*");
  std::vector<std::string> results2 = ctx_interface.ContextQuery(".*", "blob_[0-9]+");
  std::vector<std::string> results3 = ctx_interface.ContextQuery("my_tag", "my_blob");

  // Just verify the function completes without crashing
  HLOG(kInfo, "Pattern 1 returned {} results", results1.size());
  HLOG(kInfo, "Pattern 2 returned {} results", results2.size());
  HLOG(kInfo, "Pattern 3 returned {} results", results3.size());
  HLOG(kSuccess, "PASSED: Specific patterns test");
}

int main(int argc, char** argv) {
  (void)argc;  // Suppress unused parameter warning
  (void)argv;  // Suppress unused parameter warning

  HLOG(kInfo, "========================================");
  HLOG(kInfo, "ContextInterface::ContextQuery Tests");
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
    test_basic_query();

    test_specific_patterns();

    HLOG(kSuccess, "All tests PASSED!");
    return 0;
  } catch (const std::exception& e) {
    HLOG(kError, "Test FAILED with exception: {}", e.what());
    return 1;
  }
}
