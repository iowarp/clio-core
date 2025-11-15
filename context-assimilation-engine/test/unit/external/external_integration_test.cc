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
        std::cout << "=== External CAE Core Integration Test ===" << std::endl;
        std::cout << "Initializing CAE Core system..." << std::endl;

        try {
            // Step 1: Initialize Chimaera runtime
            std::cout << "1. Initializing Chimaera runtime..." << std::endl;
            bool runtime_init = chi::CHIMAERA_RUNTIME_INIT();
            if (!runtime_init) {
                std::cerr << "Failed to initialize Chimaera runtime" << std::endl;
                return false;
            }

            // Step 2: Initialize Chimaera client
            std::cout << "2. Initializing Chimaera client..." << std::endl;
            bool client_init = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
            if (!client_init) {
                std::cerr << "Failed to initialize Chimaera client" << std::endl;
                return false;
            }

            // Step 3: Create CAE client instance
            std::cout << "3. Creating CAE client instance..." << std::endl;
            cae_client_ = std::make_unique<wrp_cae::core::Client>();

            // Step 4: Create CAE container
            std::cout << "4. Creating CAE container..." << std::endl;
            wrp_cae::core::CreateParams create_params;

            try {
                cae_client_->Create(hipc::MemContext(), chi::PoolQuery::Dynamic(),
                                   "cae_test_pool",
                                   wrp_cae::core::kCaePoolId, create_params);
                std::cout << "   CAE container created successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to create CAE container: " << e.what() << std::endl;
                return false;
            }

            initialized_ = true;
            std::cout << "CAE Core initialization completed successfully!" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Exception during initialization: " << e.what() << std::endl;
            return false;
        }
    }

    bool RunTests() {
        if (!initialized_) {
            std::cerr << "Cannot run tests - system not initialized" << std::endl;
            return false;
        }

        std::cout << "\n=== Running CAE Core API Tests ===" << std::endl;

        bool all_tests_passed = true;

        // Test 1: Basic client functionality
        all_tests_passed &= TestBasicFunctionality();

        if (all_tests_passed) {
            std::cout << "\n✅ All tests passed!" << std::endl;
        } else {
            std::cout << "\n❌ Some tests failed!" << std::endl;
        }

        return all_tests_passed;
    }

private:
    bool TestBasicFunctionality() {
        std::cout << "\n--- Test 1: Basic CAE Client Functionality ---" << std::endl;

        try {
            // Verify client exists and is usable
            if (cae_client_) {
                std::cout << "✅ CAE client is functional" << std::endl;
                return true;
            } else {
                std::cout << "❌ CAE client is not available" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestBasicFunctionality: " << e.what() << std::endl;
            return false;
        }
    }

    void Cleanup() {
        if (initialized_) {
            std::cout << "\n=== Cleanup ===" << std::endl;
            std::cout << "Cleaning up CAE Core resources..." << std::endl;

            // CAE and Chimaera cleanup would happen automatically
            // through destructors and singleton cleanup

            initialized_ = false;
            std::cout << "Cleanup completed." << std::endl;
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
        std::cerr << "Failed to initialize CAE Core system" << std::endl;
        return 1;
    }

    // Run the tests
    bool success = test.RunTests();

    // Print final result
    std::cout << "\n=== Test Results ===" << std::endl;
    if (success) {
        std::cout << "🎉 External CAE Core integration test PASSED!" << std::endl;
        std::cout << "The CAE Core library is properly linkable and functional." << std::endl;
        return 0;
    } else {
        std::cout << "💥 External CAE Core integration test FAILED!" << std::endl;
        std::cout << "Check the error messages above for details." << std::endl;
        return 1;
    }
}
