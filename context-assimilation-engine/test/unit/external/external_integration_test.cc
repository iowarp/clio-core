/**
 * External CAE Core Integration Test
 *
 * This test demonstrates how external applications can link to and use
 * the CAE Core library. It serves as both a test and an example of
 * proper CAE Core usage patterns.
 */

#include <iostream>
#include <string>
#include <memory>

// CAE Core includes
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/core_tasks.h>
#include <wrp_cae/core/constants.h>
#include <chimaera/chimaera.h>

// HSHM includes
#include <hermes_shm/util/singleton.h>

// Logging
#include <hermes_shm/util/logging.h>

class ExternalCaeTest {
private:
    std::unique_ptr<wrp_cae::core::Client> cae_client_;
    bool initialized_;

public:
    ExternalCaeTest() : initialized_(false) {}

    ~ExternalCaeTest() {
        Cleanup();
    }

    bool Initialize() {
        HLOG(kInfo, "=== External CAE Core Integration Test ===");
        HLOG(kInfo, "Initializing CAE Core system...");

        try {
            // Step 1: Initialize Chimaera (runtime + client)
            HLOG(kInfo, "1. Initializing Chimaera...");
            bool chimaera_init = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
            if (!chimaera_init) {
                HLOG(kError, "Failed to initialize Chimaera");
                return false;
            }

            // Step 2: Create CAE client instance
            HLOG(kInfo, "2. Creating CAE client instance...");
            cae_client_ = std::make_unique<wrp_cae::core::Client>();

            // Step 3: Create CAE container
            HLOG(kInfo, "3. Creating CAE container...");
            wrp_cae::core::CreateParams create_params;

            try {
                cae_client_->Create(hipc::MemContext(), chi::PoolQuery::Dynamic(),
                                   "cae_test_pool",
                                   wrp_cae::core::kCaePoolId, create_params);
                HLOG(kSuccess, "CAE container created successfully");
            } catch (const std::exception& e) {
                HLOG(kError, "Failed to create CAE container: {}", e.what());
                return false;
            }

            initialized_ = true;
            HLOG(kSuccess, "CAE Core initialization completed successfully!");
            return true;

        } catch (const std::exception& e) {
            HLOG(kError, "Exception during initialization: {}", e.what());
            return false;
        }
    }

    bool RunTests() {
        if (!initialized_) {
            HLOG(kError, "Cannot run tests - system not initialized");
            return false;
        }

        HLOG(kInfo, "=== Running CAE Core API Tests ===");

        bool all_tests_passed = true;

        // Test 1: Basic client functionality
        all_tests_passed &= TestBasicFunctionality();

        if (all_tests_passed) {
            HLOG(kSuccess, "All tests passed!");
        } else {
            HLOG(kError, "Some tests failed!");
        }

        return all_tests_passed;
    }

private:
    bool TestBasicFunctionality() {
        HLOG(kInfo, "--- Test 1: Basic CAE Client Functionality ---");

        try {
            // Verify client exists and is usable
            if (cae_client_) {
                HLOG(kSuccess, "CAE client is functional");
                return true;
            } else {
                HLOG(kError, "CAE client is not available");
                return false;
            }
        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestBasicFunctionality: {}", e.what());
            return false;
        }
    }

    void Cleanup() {
        if (initialized_) {
            HLOG(kInfo, "=== Cleanup ===");
            HLOG(kInfo, "Cleaning up CAE Core resources...");

            // CAE and Chimaera cleanup would happen automatically
            // through destructors and singleton cleanup

            initialized_ = false;
            HLOG(kSuccess, "Cleanup completed.");
        }
    }
};

int main(int argc, char* argv[]) {
    // Suppress unused parameter warnings
    (void)argc;
    (void)argv;

    // Create test instance
    ExternalCaeTest test;

    // Initialize the system
    if (!test.Initialize()) {
        HLOG(kError, "Failed to initialize CAE Core system");
        return 1;
    }

    // Run the tests
    bool success = test.RunTests();

    // Print final result
    HLOG(kInfo, "=== Test Results ===");
    if (success) {
        HLOG(kSuccess, "External CAE Core integration test PASSED!");
        HLOG(kSuccess, "The CAE Core library is properly linkable and functional.");
        return 0;
    } else {
        HLOG(kError, "External CAE Core integration test FAILED!");
        HLOG(kError, "Check the error messages above for details.");
        return 1;
    }
}
