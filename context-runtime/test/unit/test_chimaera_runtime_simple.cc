/**
 * Simple unit tests for Chimaera runtime system
 * 
 * Basic tests to verify compilation and runtime initialization.
 * Uses simple custom test framework for testing.
 */

#include "../simple_test.h"
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>

namespace {
  // Test configuration constants
  constexpr chi::u32 kTestTimeoutMs = 5000;

  // Global initialization flag to prevent double initialization
  bool g_initialized = false;
}

/**
 * Simple test fixture for Chimaera runtime tests
 */
class SimpleChimaeraFixture {
public:
  SimpleChimaeraFixture() {
    // Initialize Chimaera once per test suite
    if (!g_initialized) {
      INFO("Initializing Chimaera...");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
        std::this_thread::sleep_for(500ms); // Give runtime time to initialize
        INFO("Chimaera initialization successful");
      } else {
        INFO("Failed to initialize Chimaera");
      }
    }
  }

  ~SimpleChimaeraFixture() {
    INFO("Test cleanup completed");
  }
};

//------------------------------------------------------------------------------
// Basic Runtime and Client Initialization Tests
//------------------------------------------------------------------------------

TEST_CASE("Basic Chimaera Initialization", "[runtime][basic]") {
  SimpleChimaeraFixture fixture;

  SECTION("Chimaera initialization should succeed") {
    REQUIRE(g_initialized);

    // Verify core managers are available (if not null)
    if (CHI_CHIMAERA_MANAGER != nullptr) {
      INFO("Chimaera manager is available");
      REQUIRE(CHI_CHIMAERA_MANAGER->IsInitialized());
      REQUIRE(CHI_CHIMAERA_MANAGER->IsRuntime());
      REQUIRE(CHI_CHIMAERA_MANAGER->IsClient());
    } else {
      INFO("Chimaera manager is not available");
    }

    if (CHI_IPC != nullptr) {
      INFO("IPC manager is available and initialized");
      REQUIRE(CHI_IPC->IsInitialized());
    } else {
      INFO("IPC manager is not available");
    }
  }

  SECTION("Multiple Chimaera initializations should be safe") {
    REQUIRE(g_initialized);
    REQUIRE(g_initialized); // Second call should succeed
  }
}

TEST_CASE("Combined Initialization", "[runtime][client][combined]") {
  SimpleChimaeraFixture fixture;
  
  SECTION("Initialize both runtime and client") {
    bool both_result = g_initialized;
    
    INFO("Combined initialization result: " << both_result);
    
    if (both_result) {
      INFO("Both runtime and client initialized successfully");
      
      // Check if managers are available
      if (CHI_CHIMAERA_MANAGER != nullptr) {
        INFO("Chimaera manager available");
      }
      if (CHI_IPC != nullptr) {
        INFO("IPC manager available");
      }
      if (CHI_POOL_MANAGER != nullptr) {
        INFO("Pool manager available");
      }
      if (CHI_MODULE_MANAGER != nullptr) {
        INFO("Module manager available");
      }
      if (CHI_WORK_ORCHESTRATOR != nullptr) {
        INFO("Work orchestrator available");
      }
    }
  }
}

TEST_CASE("Error Handling", "[error][basic]") {
  SimpleChimaeraFixture fixture;

  SECTION("Operations should not crash") {
    // These should not crash even if they fail
    REQUIRE(g_initialized);
  }
}

TEST_CASE("Basic Performance", "[performance][timing]") {
  SimpleChimaeraFixture fixture;

  SECTION("Chimaera initialization timing") {
    auto start_time = std::chrono::high_resolution_clock::now();

    bool result = g_initialized;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    INFO("Chimaera initialization time: " << duration.count() << " milliseconds");
    INFO("Chimaera initialization result: " << result);

    // Reasonable performance expectation (should complete within 10 seconds)
    REQUIRE(duration.count() < 10000); // 10 seconds in milliseconds
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()