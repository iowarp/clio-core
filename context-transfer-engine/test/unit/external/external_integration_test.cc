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

// Logging
#include <hermes_shm/util/logging.h>

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
        HLOG(kInfo, "=== External CTE Core Integration Test ===");
        HLOG(kInfo, "Initializing CTE Core system...");

        try {
            // Step 1: Initialize Chimaera (runtime + client)
            HLOG(kInfo, "1. Initializing Chimaera...");
            bool chimaera_init = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
            if (!chimaera_init) {
                HLOG(kError, "Failed to initialize Chimaera");
                return false;
            }

            // Step 2: Initialize CTE subsystem
            HLOG(kInfo, "2. Initializing CTE subsystem...");
            bool cte_init = wrp_cte::core::WRP_CTE_CLIENT_INIT();
            if (!cte_init) {
                HLOG(kError, "Failed to initialize CTE subsystem");
                return false;
            }

            // Step 4: Get CTE client instance
            HLOG(kInfo, "4. Getting CTE client instance...");
            auto *global_client = WRP_CTE_CLIENT;
            if (!global_client) {
                HLOG(kError, "Failed to get CTE client instance");
                return false;
            }

            // Create our own client instance for testing
            cte_client_ = std::make_unique<wrp_cte::core::Client>();

            // Step 5: Create CTE container
            HLOG(kInfo, "5. Creating CTE container...");
            wrp_cte::core::CreateParams create_params;

            try{
                // Use CTE Core constants from core_tasks.h
                cte_client_->Create(hipc::MemContext(), chi::PoolQuery::Dynamic(),
                                   wrp_cte::core::kCtePoolName,
                                   wrp_cte::core::kCtePoolId, create_params);
                HLOG(kInfo, "CTE container created successfully");
            } catch (const std::exception& e) {
                HLOG(kError, "Failed to create CTE container: {}", e.what());
                return false;
            }

            initialized_ = true;
            HLOG(kSuccess, "CTE Core initialization completed successfully!");
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

        HLOG(kInfo, "\n=== Running CTE Core API Tests ===");

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
            HLOG(kSuccess, "\nAll tests passed!");
        } else {
            HLOG(kError, "\nSome tests failed!");
        }

        return all_tests_passed;
    }

private:
    bool TestRegisterTarget() {
        HLOG(kInfo, "\n--- Test 1: Register Storage Target ---");

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
                HLOG(kSuccess, "Storage target registered successfully");
                return true;
            } else {
                HLOG(kError, "Failed to register storage target (code: {})", result);
                return false;
            }
        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestRegisterTarget: {}", e.what());
            return false;
        }
    }

    bool TestCreateTagAndBlob() {
        HLOG(kInfo, "\n--- Test 2: Create Tag and Store Blob ---");

        try {
            // Create test data
            std::vector<char> test_data(kTestDataSize);
            for (size_t i = 0; i < kTestDataSize; ++i) {
                test_data[i] = static_cast<char>(i % 256);
            }

            // Allocate shared memory for the data
            hipc::FullPtr<char> shared_data = CHI_IPC->AllocateBuffer(kTestDataSize);
            if (shared_data.IsNull()) {
                HLOG(kError, "Failed to allocate shared memory buffer");
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

            HLOG(kSuccess, "Tag created/retrieved with ID: {}", tag_id);

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
                HLOG(kSuccess, "Blob stored successfully");
                return true;
            } else {
                HLOG(kError, "Failed to store blob");
                return false;
            }

        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestCreateTagAndBlob: {}", e.what());
            return false;
        }
    }

    bool TestRetrieveBlob() {
        HLOG(kInfo, "\n--- Test 3: Retrieve Blob Data ---");

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
                HLOG(kError, "Failed to allocate read buffer");
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
                    HLOG(kSuccess, "Blob retrieved and data verified successfully");
                    return true;
                } else {
                    HLOG(kError, "Data verification failed");
                    return false;
                }
            } else {
                HLOG(kError, "Failed to retrieve blob");
                return false;
            }

        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestRetrieveBlob: {}", e.what());
            return false;
        }
    }

    bool TestTelemetryCollection() {
        HLOG(kInfo, "\n--- Test 4: Test Telemetry Collection ---");

        try {
            // Poll telemetry log
            std::vector<wrp_cte::core::CteTelemetry> telemetry =
                cte_client_->PollTelemetryLog(hipc::MemContext(), 0);

            HLOG(kSuccess, "Retrieved {} telemetry entries", telemetry.size());

            // Display first few entries
            size_t display_count = std::min(static_cast<size_t>(3), telemetry.size());
            for (size_t i = 0; i < display_count; ++i) {
                const auto& entry = telemetry[i];
                HLOG(kInfo, "Entry {}: op={}, size={}, logical_time={}", i,
                     static_cast<int>(entry.op_), entry.size_, entry.logical_time_);
            }

            return true;

        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestTelemetryCollection: {}", e.what());
            return false;
        }
    }

    bool TestListTargets() {
        HLOG(kInfo, "\n--- Test 5: List Storage Targets ---");

        try {
            std::vector<std::string> targets =
                cte_client_->ListTargets(hipc::MemContext());

            HLOG(kSuccess, "Found {} registered targets", targets.size());

            for (size_t i = 0; i < targets.size(); ++i) {
                const auto& target_name = targets[i];
                HLOG(kInfo, "Target {}: {}", i, target_name);
            }

            return true;

        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestListTargets: {}", e.what());
            return false;
        }
    }

    bool TestGetTagSize() {
        HLOG(kInfo, "\n--- Test 6: Get Tag Size ---");

        try {
            // Get the tag ID first
            wrp_cte::core::TagId tag_id = cte_client_->GetOrCreateTag(
                hipc::MemContext(),
                kTestTagName,
                wrp_cte::core::TagId::GetNull()
            );

            size_t tag_size = cte_client_->GetTagSize(hipc::MemContext(), tag_id);

            HLOG(kSuccess, "Tag size: {} bytes", tag_size);

            // Verify it matches our expected size
            if (tag_size >= kTestDataSize) {
                HLOG(kSuccess, "Tag size verification passed");
                return true;
            } else {
                HLOG(kError, "Tag size verification failed (expected >= {}, got {})",
                     kTestDataSize, tag_size);
                return false;
            }

        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestGetTagSize: {}", e.what());
            return false;
        }
    }

    bool TestCleanupOperations() {
        HLOG(kInfo, "\n--- Test 7: Cleanup Operations ---");

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
                HLOG(kSuccess, "Blob deleted successfully");
            } else {
                HLOG(kWarning, "Blob deletion failed (may not exist)");
            }

            // Delete the tag
            bool del_tag_result = cte_client_->DelTag(hipc::MemContext(), kTestTagName);

            if (del_tag_result) {
                HLOG(kSuccess, "Tag deleted successfully");
            } else {
                HLOG(kWarning, "Tag deletion failed (may not exist)");
            }

            return true;

        } catch (const std::exception& e) {
            HLOG(kError, "Exception in TestCleanupOperations: {}", e.what());
            return false;
        }
    }

    void Cleanup() {
        if (initialized_) {
            HLOG(kInfo, "\n=== Cleanup ===");
            HLOG(kInfo, "Cleaning up CTE Core resources...");

            // CTE and Chimaera cleanup would happen automatically
            // through destructors and singleton cleanup

            initialized_ = false;
            HLOG(kInfo, "Cleanup completed.");
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
        HLOG(kError, "Failed to initialize CTE Core system");
        return 1;
    }

    // Run the tests
    bool success = test.RunTests();

    // Print final result
    HLOG(kInfo, "\n=== Test Results ===");
    if (success) {
        HLOG(kSuccess, "External CTE Core integration test PASSED!");
        HLOG(kInfo, "The CTE Core library is properly linkable and functional.");
        return 0;
    } else {
        HLOG(kError, "External CTE Core integration test FAILED!");
        HLOG(kInfo, "Check the error messages above for details.");
        return 1;
    }
}