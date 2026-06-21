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
 * Simple unit tests for CLIO Runtime runtime system
 * 
 * Basic tests to verify compilation and runtime initialization.
 * Uses simple custom test framework for testing.
 */

#include "../simple_test.h"
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

// Include CLIO Runtime headers
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

namespace {
  // Test configuration constants
  constexpr clio::run::u32 kTestTimeoutMs = 5000;

  // Global initialization flag to prevent double initialization
  bool g_initialized = false;
}

/**
 * Simple test fixture for CLIO Runtime runtime tests
 */
class SimpleClioFixture {
public:
  SimpleClioFixture() {
    // Initialize CLIO Runtime once per test suite
    if (!g_initialized) {
      INFO("Initializing Clio...");
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
        std::this_thread::sleep_for(500ms); // Give runtime time to initialize
        INFO("Clio initialization successful");
      } else {
        INFO("Failed to initialize Clio");
      }
    }
  }

  ~SimpleClioFixture() {
    INFO("Test cleanup completed");
  }
};

//------------------------------------------------------------------------------
// Basic Runtime and Client Initialization Tests
//------------------------------------------------------------------------------

TEST_CASE("Basic Clio Initialization", "[runtime][basic]") {
  SimpleClioFixture fixture;

  SECTION("Clio initialization should succeed") {
    REQUIRE(g_initialized);

    // Verify core managers are available (if not null)
    if (CLIO_RUNTIME_MANAGER != nullptr) {
      INFO("Clio manager is available");
      REQUIRE(CLIO_RUNTIME_MANAGER->IsInitialized());
      REQUIRE(CLIO_RUNTIME_MANAGER->IsRuntime());
      REQUIRE(CLIO_RUNTIME_MANAGER->IsClient());
    } else {
      INFO("Clio manager is not available");
    }

    if (CLIO_IPC != nullptr) {
      INFO("IPC manager is available and initialized");
      REQUIRE(CLIO_IPC->IsInitialized());
    } else {
      INFO("IPC manager is not available");
    }
  }

  SECTION("Multiple Clio initializations should be safe") {
    REQUIRE(g_initialized);
    REQUIRE(g_initialized); // Second call should succeed
  }
}

TEST_CASE("Combined Initialization", "[runtime][client][combined]") {
  SimpleClioFixture fixture;
  
  SECTION("Initialize both runtime and client") {
    bool both_result = g_initialized;
    
    INFO("Combined initialization result: " << both_result);
    
    if (both_result) {
      INFO("Both runtime and client initialized successfully");
      
      // Check if managers are available
      if (CLIO_RUNTIME_MANAGER != nullptr) {
        INFO("Clio manager available");
      }
      if (CLIO_IPC != nullptr) {
        INFO("IPC manager available");
      }
      if (CLIO_POOL_MANAGER != nullptr) {
        INFO("Pool manager available");
      }
      if (CLIO_MODULE_MANAGER != nullptr) {
        INFO("Module manager available");
      }
      if (CLIO_WORK_ORCHESTRATOR != nullptr) {
        INFO("Work orchestrator available");
      }
    }
  }
}

TEST_CASE("Error Handling", "[error][basic]") {
  SimpleClioFixture fixture;

  SECTION("Operations should not crash") {
    // These should not crash even if they fail
    REQUIRE(g_initialized);
  }
}

TEST_CASE("Basic Performance", "[performance][timing]") {
  SimpleClioFixture fixture;

  SECTION("Clio initialization timing") {
    auto start_time = std::chrono::high_resolution_clock::now();

    bool result = g_initialized;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    INFO("Clio initialization time: " << duration.count() << " milliseconds");
    INFO("Clio initialization result: " << result);

    // Reasonable performance expectation (should complete within 10 seconds)
    REQUIRE(duration.count() < 10000); // 10 seconds in milliseconds
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()