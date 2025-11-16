/**
 * External CTE Core Integration Test
 * 
 * This test demonstrates how external applications can link to and use 
 * the CTE Core library. It serves as both a test and an example of 
 * proper CTE Core usage patterns.
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>

// CTE Core includes
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/core/content_transfer_engine.h>
#include <chimaera/chimaera.h>

// Adapter includes for testing filesystem integration
#include <adapter/cae_config.h>

// HSHM includes
#include <hermes_shm/util/singleton.h>

namespace {
    constexpr size_t kTestDataSize = 1024;  // 1KB test data
    constexpr const char* kTestTagName = "external_test_tag";
    constexpr const char* kTestBlobName = "external_test_blob";
}

class ExternalCteTest {
private:
    std::unique_ptr<wrp_cte::core::Client> cte_client_;
    bool initialized_;

public:
    ExternalCteTest() : initialized_(false) {}

    ~ExternalCteTest() {
        Cleanup();
    }

    bool Initialize() {
        std::cout << "=== External CTE Core Integration Test ===" << std::endl;
        std::cout << "Initializing CTE Core system..." << std::endl;

        try {
            // Step 1: Initialize Chimaera (runtime + client)
            std::cout << "1. Initializing Chimaera..." << std::endl;
            bool chimaera_init = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
            if (!chimaera_init) {
                std::cerr << "Failed to initialize Chimaera" << std::endl;
                return false;
            }

            // Step 2: Initialize CTE subsystem
            std::cout << "2. Initializing CTE subsystem..." << std::endl;
            bool cte_init = wrp_cte::core::WRP_CTE_CLIENT_INIT();
            if (!cte_init) {
                std::cerr << "Failed to initialize CTE subsystem" << std::endl;
                return false;
            }

            // Step 4: Get CTE client instance
            std::cout << "4. Getting CTE client instance..." << std::endl;
            auto *global_client = WRP_CTE_CLIENT;
            if (!global_client) {
                std::cerr << "Failed to get CTE client instance" << std::endl;
                return false;
            }

            // Create our own client instance for testing
            cte_client_ = std::make_unique<wrp_cte::core::Client>();

            // Step 5: Create CTE container
            std::cout << "5. Creating CTE container..." << std::endl;
            wrp_cte::core::CreateParams create_params;

            try{
                // Use CTE Core constants from core_tasks.h
                cte_client_->Create(hipc::MemContext(), chi::PoolQuery::Dynamic(),
                                   wrp_cte::core::kCtePoolName,
                                   wrp_cte::core::kCtePoolId, create_params);
                std::cout << "   CTE container created successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to create CTE container: " << e.what() << std::endl;
                return false;
            }

            initialized_ = true;
            std::cout << "CTE Core initialization completed successfully!" << std::endl;
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

        std::cout << "\n=== Running CTE Core API Tests ===" << std::endl;

        bool all_tests_passed = true;

        // Test 1: Register a storage target
        all_tests_passed &= TestRegisterTarget();

        // Test 2: Create tag and store blob
        all_tests_passed &= TestCreateTagAndBlob();

        // Test 3: Retrieve blob data
        all_tests_passed &= TestRetrieveBlob();

        // Test 4: Test telemetry collection
        all_tests_passed &= TestTelemetryCollection();

        // Test 5: List targets
        all_tests_passed &= TestListTargets();

        // Test 6: Get tag size
        all_tests_passed &= TestGetTagSize();

        // Test 7: Cleanup operations
        all_tests_passed &= TestCleanupOperations();

        if (all_tests_passed) {
            std::cout << "\n✅ All tests passed!" << std::endl;
        } else {
            std::cout << "\n❌ Some tests failed!" << std::endl;
        }

        return all_tests_passed;
    }

private:
    bool TestRegisterTarget() {
        std::cout << "\n--- Test 1: Register Storage Target ---" << std::endl;
        
        try {
            std::string target_name = "/tmp/cte_external_test_target";
            chi::u64 target_size = 100 * 1024 * 1024;  // 100MB

            chi::u32 result = cte_client_->RegisterTarget(
                hipc::MemContext(), 
                target_name,
                chimaera::bdev::BdevType::kFile,
                target_size
            );

            if (result == 0) {
                std::cout << "✅ Storage target registered successfully" << std::endl;
                return true;
            } else {
                std::cout << "❌ Failed to register storage target (code: " << result << ")" << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestRegisterTarget: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestCreateTagAndBlob() {
        std::cout << "\n--- Test 2: Create Tag and Store Blob ---" << std::endl;

        try {
            // Create test data
            std::vector<char> test_data(kTestDataSize);
            for (size_t i = 0; i < kTestDataSize; ++i) {
                test_data[i] = static_cast<char>(i % 256);
            }

            // Allocate shared memory for the data
            hipc::FullPtr<char> shared_data = CHI_IPC->AllocateBuffer(kTestDataSize);
            if (shared_data.IsNull()) {
                std::cout << "❌ Failed to allocate shared memory buffer" << std::endl;
                return false;
            }

            // Copy data to shared memory
            memcpy(shared_data.ptr_, test_data.data(), kTestDataSize);

            // Create or get tag
            wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(
                hipc::MemContext(),
                kTestTagName,
                wrp_cte::core::TagId::GetNull()
            );

            std::cout << "✅ Tag created/retrieved with ID: " << tag_id << std::endl;

            // Store blob data
            bool put_result = cte_client_->PutBlob(
                hipc::MemContext(),
                tag_id,
                kTestBlobName,
                0,  // offset
                kTestDataSize,
                shared_data.shm_,  // Use .shm_ to get the Pointer
                0.8f,  // score
                0  // flags
            );

            if (put_result) {
                std::cout << "✅ Blob stored successfully" << std::endl;
                return true;
            } else {
                std::cout << "❌ Failed to store blob" << std::endl;
                return false;
            }

        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestCreateTagAndBlob: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestRetrieveBlob() {
        std::cout << "\n--- Test 3: Retrieve Blob Data ---" << std::endl;

        try {
            // Get the tag ID first
            wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(
                hipc::MemContext(),
                kTestTagName,
                wrp_cte::core::TagId::GetNull()
            );

            // Allocate buffer for reading
            hipc::FullPtr<char> read_buffer = CHI_IPC->AllocateBuffer(kTestDataSize);
            if (read_buffer.IsNull()) {
                std::cout << "❌ Failed to allocate read buffer" << std::endl;
                return false;
            }

            // Retrieve blob data
            bool get_result = cte_client_->GetBlob(
                hipc::MemContext(),
                tag_id,
                kTestBlobName,
                0,  // offset
                kTestDataSize,
                0,  // flags
                read_buffer.shm_
            );

            if (get_result) {
                // Verify data integrity
                const char* read_data = static_cast<const char*>(read_buffer.ptr_);
                bool data_matches = true;
                
                for (size_t i = 0; i < kTestDataSize; ++i) {
                    if (read_data[i] != static_cast<char>(i % 256)) {
                        data_matches = false;
                        break;
                    }
                }

                if (data_matches) {
                    std::cout << "✅ Blob retrieved and data verified successfully" << std::endl;
                    return true;
                } else {
                    std::cout << "❌ Data verification failed" << std::endl;
                    return false;
                }
            } else {
                std::cout << "❌ Failed to retrieve blob" << std::endl;
                return false;
            }

        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestRetrieveBlob: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestTelemetryCollection() {
        std::cout << "\n--- Test 4: Test Telemetry Collection ---" << std::endl;

        try {
            // Poll telemetry log
            std::vector<wrp_cte::core::CteTelemetry> telemetry = 
                cte_client_->PollTelemetryLog(hipc::MemContext(), 0);

            std::cout << "✅ Retrieved " << telemetry.size() << " telemetry entries" << std::endl;

            // Display first few entries
            size_t display_count = std::min(static_cast<size_t>(3), telemetry.size());
            for (size_t i = 0; i < display_count; ++i) {
                const auto& entry = telemetry[i];
                std::cout << "   Entry " << i << ": op=" << static_cast<int>(entry.op_)
                          << ", size=" << entry.size_ 
                          << ", logical_time=" << entry.logical_time_ << std::endl;
            }

            return true;

        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestTelemetryCollection: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestListTargets() {
        std::cout << "\n--- Test 5: List Storage Targets ---" << std::endl;

        try {
            std::vector<std::string> targets =
                cte_client_->ListTargets(hipc::MemContext());

            std::cout << "✅ Found " << targets.size() << " registered targets" << std::endl;

            for (size_t i = 0; i < targets.size(); ++i) {
                const auto& target_name = targets[i];
                std::cout << "   Target " << i << ": " << target_name << std::endl;
            }

            return true;

        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestListTargets: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestGetTagSize() {
        std::cout << "\n--- Test 6: Get Tag Size ---" << std::endl;

        try {
            // Get the tag ID first
            wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(
                hipc::MemContext(),
                kTestTagName,
                wrp_cte::core::TagId::GetNull()
            );

            size_t tag_size = cte_client_->GetTagSize(hipc::MemContext(), tag_id);

            std::cout << "✅ Tag size: " << tag_size << " bytes" << std::endl;

            // Verify it matches our expected size
            if (tag_size >= kTestDataSize) {
                std::cout << "✅ Tag size verification passed" << std::endl;
                return true;
            } else {
                std::cout << "❌ Tag size verification failed (expected >= " 
                          << kTestDataSize << ", got " << tag_size << ")" << std::endl;
                return false;
            }

        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestGetTagSize: " << e.what() << std::endl;
            return false;
        }
    }

    bool TestCleanupOperations() {
        std::cout << "\n--- Test 7: Cleanup Operations ---" << std::endl;

        try {
            // Get the tag ID
            wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(
                hipc::MemContext(),
                kTestTagName,
                wrp_cte::core::TagId::GetNull()
            );

            // Delete the blob
            bool del_blob_result = cte_client_->DelBlob(
                hipc::MemContext(),
                tag_id,
                kTestBlobName
            );

            if (del_blob_result) {
                std::cout << "✅ Blob deleted successfully" << std::endl;
            } else {
                std::cout << "⚠️  Blob deletion failed (may not exist)" << std::endl;
            }

            // Delete the tag
            bool del_tag_result = cte_client_->DelTag(hipc::MemContext(), kTestTagName);

            if (del_tag_result) {
                std::cout << "✅ Tag deleted successfully" << std::endl;
            } else {
                std::cout << "⚠️  Tag deletion failed (may not exist)" << std::endl;
            }

            return true;

        } catch (const std::exception& e) {
            std::cout << "❌ Exception in TestCleanupOperations: " << e.what() << std::endl;
            return false;
        }
    }

    void Cleanup() {
        if (initialized_) {
            std::cout << "\n=== Cleanup ===" << std::endl;
            std::cout << "Cleaning up CTE Core resources..." << std::endl;
            
            // CTE and Chimaera cleanup would happen automatically
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
    ExternalCteTest test;

    // Initialize the system
    if (!test.Initialize()) {
        std::cerr << "Failed to initialize CTE Core system" << std::endl;
        return 1;
    }

    // Run the tests
    bool success = test.RunTests();

    // Print final result
    std::cout << "\n=== Test Results ===" << std::endl;
    if (success) {
        std::cout << "🎉 External CTE Core integration test PASSED!" << std::endl;
        std::cout << "The CTE Core library is properly linkable and functional." << std::endl;
        return 0;
    } else {
        std::cout << "💥 External CTE Core integration test FAILED!" << std::endl;
        std::cout << "Check the error messages above for details." << std::endl;
        return 1;
    }
}