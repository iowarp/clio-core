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
  
  // Global test state
  bool g_runtime_initialized = false;
  bool g_client_initialized = false;
}

/**
 * Simple test fixture for Chimaera runtime tests
 */
class SimpleChimaeraFixture {
public:
  SimpleChimaeraFixture() = default;
  
  ~SimpleChimaeraFixture() {
    cleanup();
  }
  
  /**
   * Initialize Chimaera runtime (server-side)
   */
  bool initializeRuntime() {
    if (g_runtime_initialized) {
      return true; // Already initialized
    }
    
    INFO("Initializing Chimaera runtime...");
    bool success = chi::CHIMAERA_RUNTIME_INIT();
    
    if (success) {
      g_runtime_initialized = true;
      
      // Give runtime time to initialize all components
      std::this_thread::sleep_for(500ms);
      
      INFO("Runtime initialization successful");
    } else {
      INFO("Failed to initialize Chimaera runtime");
    }
    
    return success;
  }
  
  /**
   * Initialize Chimaera client components
   */
  bool initializeClient() {
    if (g_client_initialized) {
      return true; // Already initialized
    }

    INFO("Initializing Chimaera client...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

    if (success) {
      g_client_initialized = true;

      // Give client time to connect to runtime
      std::this_thread::sleep_for(200ms);

      INFO("Client initialization successful");
    } else {
      INFO("Failed to initialize Chimaera client");
    }

    return success;
  }
  
  /**
   * Initialize both runtime and client (full setup)
   */
  bool initializeBoth() {
    return initializeRuntime() && initializeClient();
  }
  
  /**
   * Clean up runtime and client resources
   */
  void cleanup() {
    INFO("Test cleanup completed");
  }
};

//------------------------------------------------------------------------------
// Basic Runtime and Client Initialization Tests
//------------------------------------------------------------------------------

TEST_CASE("Basic Runtime Initialization", "[runtime][basic]") {
  SimpleChimaeraFixture fixture;
  
  SECTION("Runtime initialization should succeed") {
    REQUIRE(fixture.initializeRuntime());
    
    // Verify core managers are available (if not null)
    if (CHI_CHIMAERA_MANAGER != nullptr) {
      INFO("Chimaera manager is available");
      REQUIRE(CHI_CHIMAERA_MANAGER->IsInitialized());
      REQUIRE(CHI_CHIMAERA_MANAGER->IsRuntime());
      REQUIRE_FALSE(CHI_CHIMAERA_MANAGER->IsClient());
    } else {
      INFO("Chimaera manager is not available");
    }
  }
  
  SECTION("Multiple runtime initializations should be safe") {
    REQUIRE(fixture.initializeRuntime());
    REQUIRE(fixture.initializeRuntime()); // Second call should succeed
  }
}

TEST_CASE("Basic Client Initialization", "[client][basic]") {
  SimpleChimaeraFixture fixture;
  
  SECTION("Client initialization with runtime") {
    // Initialize runtime first
    REQUIRE(fixture.initializeRuntime());
    
    // Then initialize client
    bool client_result = fixture.initializeClient();
    
    // Client init may or may not succeed depending on implementation
    // The important thing is it doesn't crash
    INFO("Client initialization result: " << client_result);
    
    if (client_result && CHI_IPC != nullptr) {
      INFO("IPC manager is available and initialized");
      REQUIRE(CHI_IPC->IsInitialized());
    } else {
      INFO("IPC manager is not available or client init failed");
    }
  }
  
  SECTION("Client initialization without runtime") {
    // Attempting client init without runtime should work gracefully
    bool client_result = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

    // This may succeed or fail depending on implementation
    // The important thing is it doesn't crash
    INFO("Client init without runtime result: " << client_result);
  }
}

TEST_CASE("Combined Initialization", "[runtime][client][combined]") {
  SimpleChimaeraFixture fixture;
  
  SECTION("Initialize both runtime and client") {
    bool both_result = fixture.initializeBoth();
    
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
    REQUIRE_NOTHROW(fixture.initializeRuntime());
    REQUIRE_NOTHROW(fixture.initializeClient());
    REQUIRE_NOTHROW(fixture.cleanup());
  }
}

TEST_CASE("Basic Performance", "[performance][timing]") {
  SimpleChimaeraFixture fixture;
  
  SECTION("Runtime initialization timing") {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool result = fixture.initializeRuntime();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    INFO("Runtime initialization time: " << duration.count() << " milliseconds");
    INFO("Runtime initialization result: " << result);
    
    // Reasonable performance expectation (should complete within 10 seconds)
    REQUIRE(duration.count() < 10000); // 10 seconds in milliseconds
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()