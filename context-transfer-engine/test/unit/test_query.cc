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
 * CTE CORE QUERY API UNIT TESTS
 *
 * This test suite provides comprehensive tests for the TagQuery and BlobQuery
 * APIs. These tests verify regex pattern matching functionality for both tags
 * and blobs in the CTE core system.
 *
 * Test Coverage:
 * 1. TagQuery with various regex patterns (exact match, wildcards, alternation)
 * 2. BlobQuery with tag and blob regex combinations
 * 3. Broadcast pool query behavior
 * 4. Empty result sets and edge cases
 *
 * Following CLAUDE.md requirements:
 * - Use simple_test.h framework (NOT Catch2 - Catch2 causes segfaults with Chimaera runtime)
 * - Use proper runtime initialization
 * - Use chi::kAdminPoolId for CreateTask operations
 * - Use semantic names for QueueIds and priorities
 * - Never use null pool queries - always use Local() or Broadcast()
 * - Follow Google C++ style guide
 */

 #ifndef _WIN32
 #include <sys/stat.h>
 #include <unistd.h>
 #endif

 #include <cstdlib>
 #include <filesystem>
 #include <thread>
 #include <chrono>
 #include <limits.h>

 #include "simple_test.h"

 using namespace std::chrono_literals;

 // Chimaera core includes
 #include <chimaera/admin/admin_tasks.h>
 #include <chimaera/bdev/bdev_client.h>
 #include <chimaera/bdev/bdev_tasks.h>
 #include <chimaera/chimaera.h>
 #include <wrp_cte/core/core_client.h>
 #include <wrp_cte/core/core_runtime.h>
 #include <wrp_cte/core/core_tasks.h>

 namespace fs = std::filesystem;

 // Global state tracking for initialization (following pattern from other tests)
 namespace {
   bool g_initialized = false;
 }


 /**
  * Test fixture for CTE Query API tests
  *
  * This fixture provides runtime initialization and sets up test data including
  * tags and blobs for query testing.
  */
 class CTEQueryTestFixture {
 public:
   // Semantic names for queue IDs and priorities (following CLAUDE.md
   // requirements)
   static constexpr chi::QueueId kCTEMainQueueId = chi::QueueId(1);
   static constexpr chi::QueueId kCTEWorkerQueueId = chi::QueueId(2);
   static constexpr chi::u32 kCTEHighPriority = 1;
   static constexpr chi::u32 kCTENormalPriority = 2;

   // Test configuration constants
   static constexpr chi::u64 kTestTargetSize = 1024ULL * 1024 * 100; // 100MB test target
   static constexpr size_t kTestBlobSize = 4096; // 4KB test blobs

   // CTE Core pool configuration - use constants from core_tasks.h
   static inline const chi::PoolId& kCTECorePoolId = wrp_cte::core::kCtePoolId;
   static inline const char* kCTECorePoolName = wrp_cte::core::kCtePoolName;

   std::string test_storage_path_;

   // Test data: tags and blobs created during setup
   std::vector<std::string> test_tags_;
   std::vector<std::pair<std::string, std::string>> test_blobs_; // (tag_name, blob_name)

   // Flag to track if setup has been completed (singleton initialization)
   bool setup_completed_ = false;

   /**
    * Helper: Async TagQuery
    */
   static std::vector<std::string> TagQueryAsync(wrp_cte::core::Client* client,
                                                  const std::string& tag_pattern,
                                                  chi::u32 flags,
                                                  const chi::PoolQuery& pool_query) {
     auto task = client->AsyncTagQuery(tag_pattern, flags, pool_query);
     task.Wait();
     return task->results_;
   }

   /**
    * Helper: Async BlobQuery
    */
   static std::vector<std::pair<std::string, std::string>> BlobQueryAsync(
       wrp_cte::core::Client* client,
       const std::string& tag_pattern,
       const std::string& blob_pattern,
       chi::u32 flags,
       const chi::PoolQuery& pool_query) {
     auto task = client->AsyncBlobQuery(tag_pattern, blob_pattern, flags, pool_query);
     task.Wait();
     // Combine tag_names_ and blob_names_ into pairs
     std::vector<std::pair<std::string, std::string>> results;
     size_t count = std::min(task->tag_names_.size(), task->blob_names_.size());
     results.reserve(count);
     for (size_t i = 0; i < count; ++i) {
       results.emplace_back(task->tag_names_[i], task->blob_names_[i]);
     }
     return results;
   }

   CTEQueryTestFixture() {
     // Initialize test storage path in home directory
     std::string home_dir = hshm::SystemInfo::Getenv("HOME");
     if (home_dir.empty()) {
       throw std::runtime_error("HOME environment variable is not set");
     }

     test_storage_path_ = home_dir + "/cte_query_test.dat";

     // Clean up any existing test file
     if (fs::exists(test_storage_path_)) {
       fs::remove(test_storage_path_);
     }

     // Initialize Chimaera and CTE client once per test suite
     if (!g_initialized) {
       bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
       if (!success) {
         throw std::runtime_error("Failed to initialize Chimaera runtime");
       }

       // Wait for runtime to be fully ready
       std::this_thread::sleep_for(500ms);

       success = wrp_cte::core::WRP_CTE_CLIENT_INIT();
       if (!success) {
         throw std::runtime_error("Failed to initialize CTE client");
       }

       g_initialized = true;
     }

     // Create test data only once (singleton pattern)
     if (!setup_completed_) {
       setupTestData();
       setup_completed_ = true;
     }
   }

   ~CTEQueryTestFixture() {
     // Clean up test storage
     if (fs::exists(test_storage_path_)) {
       fs::remove(test_storage_path_);
     }
   }

 private:
   /**
    * Setup test data: create tags and blobs for query testing
    */
   void setupTestData() {
     auto *cte_client = WRP_CTE_CLIENT;

     // Create test storage target using bdev client
     chi::PoolId bdev_pool_id(200, 0);  // Custom pool ID for bdev
     chimaera::bdev::Client bdev_client(bdev_pool_id);
     auto create_task = bdev_client.AsyncCreate(chi::PoolQuery::Dynamic(), test_storage_path_,
                                                 bdev_pool_id, chimaera::bdev::BdevType::kFile);
     create_task.Wait();

     // Wait for storage target creation
     std::this_thread::sleep_for(100ms);

     // Register the storage target with CTE
     auto reg_task = cte_client->AsyncRegisterTarget(test_storage_path_,
                                                      chimaera::bdev::BdevType::kFile,
                                                      kTestTargetSize, chi::PoolQuery::Local(), bdev_pool_id);
     reg_task.Wait();
     std::this_thread::sleep_for(100ms);

     // Create test tags and blobs
     // Tag set 1: user_data
     createTestBlobs("user_data", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});

     // Tag set 2: user_logs
     createTestBlobs("user_logs", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});

     // Tag set 3: system_config
     createTestBlobs("system_config", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});

     // Tag set 4: system_cache
     createTestBlobs("system_cache", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});

     // Tag set 5-8: additional test tags
     createTestBlobs("data_archive", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});
     createTestBlobs("temp_storage", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});
     createTestBlobs("backup_data", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});
     createTestBlobs("cache_files", {"blob_001.dat", "blob_002.dat", "file_a.txt", "file_b.txt"});
   }

   /**
    * Helper function to create test blobs under a tag
    */
   void createTestBlobs(const std::string& tag_name,
                        const std::vector<std::string>& blob_names) {
     test_tags_.push_back(tag_name);

     // Use Tag wrapper class for easier blob creation
     wrp_cte::core::Tag tag(tag_name);

     // Create test data buffer
     std::vector<char> test_data(kTestBlobSize, 'X');

     for (const auto& blob_name : blob_names) {
       // Put blob with tag (uses raw pointer API)
       tag.PutBlob(blob_name, test_data.data(), kTestBlobSize);

       test_blobs_.emplace_back(tag_name, blob_name);
     }

     // Small delay to ensure blobs are created
     std::this_thread::sleep_for(50ms);
   }
 };

 /**
  * Test TagQuery with exact match pattern
  */
 TEST_CASE("TagQuery - Exact Match", "[query][tagquery][exact]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning

   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::TagQueryAsync(cte_client, "user_data", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " tags");
   REQUIRE(results.size() >= 1);

   bool found = false;
   for (const auto& tag_name : results) {
     if (tag_name == "user_data") {
       found = true;
       break;
     }
   }
   REQUIRE(found);
 }

 /**
  * Test TagQuery with wildcard pattern
  */
 TEST_CASE("TagQuery - Wildcard Pattern", "[query][tagquery][wildcard]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing TagQuery with wildcard pattern");

   // Query for all tags starting with "user_"
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::TagQueryAsync(cte_client, "user_.*", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " tags");
   REQUIRE(results.size() >= 2); // Should match user_data and user_logs

   bool found_user_data = false;
   bool found_user_logs = false;
   for (const auto& tag_name : results) {
     INFO("Found tag: " << tag_name);
     if (tag_name == "user_data") {
       found_user_data = true;
     }
     if (tag_name == "user_logs") {
       found_user_logs = true;
     }
   }

   REQUIRE(found_user_data);
   REQUIRE(found_user_logs);
 }

 /**
  * Test TagQuery with alternation pattern
  */
 TEST_CASE("TagQuery - Alternation Pattern", "[query][tagquery][alternation]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing TagQuery with alternation pattern");

   // Query for tags matching either "system_config" or "system_cache"
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::TagQueryAsync(cte_client, "system_(config|cache)", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " tags");
   REQUIRE(results.size() >= 2);

   bool found_config = false;
   bool found_cache = false;
   for (const auto& tag_name : results) {
     INFO("Found tag: " << tag_name);
     if (tag_name == "system_config") {
       found_config = true;
     }
     if (tag_name == "system_cache") {
       found_cache = true;
     }
   }

   REQUIRE(found_config);
   REQUIRE(found_cache);
 }

 /**
  * Test TagQuery with match-all pattern
  */
 TEST_CASE("TagQuery - Match All Pattern", "[query][tagquery][matchall]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing TagQuery with match-all pattern");

   // Query for all tags
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::TagQueryAsync(cte_client, ".*", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " tags");
   REQUIRE(results.size() >= fixture->test_tags_.size());

   // Verify all test tags are present
   for (const auto& expected_tag : fixture->test_tags_) {
     bool found = false;
     for (const auto &tag_name : results) {
       if (tag_name == expected_tag) {
         found = true;
         break;
       }
     }
     REQUIRE(found);
     if (!found) {
       INFO("Missing expected tag: " << expected_tag);
     }
   }
 }

 /**
  * Test TagQuery with no matches
  */
 TEST_CASE("TagQuery - No Matches", "[query][tagquery][nomatch]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing TagQuery with pattern that matches nothing");

   // Query for non-existent tag pattern
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::TagQueryAsync(cte_client, "nonexistent_tag_pattern_xyz", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " tags");
   REQUIRE(results.empty());
 }

 /**
  * Test BlobQuery with exact tag and blob match
  */
 TEST_CASE("BlobQuery - Exact Match", "[query][blobquery][exact]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing BlobQuery with exact match patterns");

   // Query for specific blob in specific tag
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::BlobQueryAsync(cte_client, "user_data", "blob_001\\.dat", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " blob pairs");
   REQUIRE(results.size() > 0);

   bool found = false;
   for (const auto &pair : results) {
     INFO("Found blob: " << pair.first << "/" << pair.second);
     if (pair.first == "user_data" && pair.second.find("blob_001.dat") != std::string::npos) {
       found = true;
       break;
     }
   }
   REQUIRE(found);
 }

 /**
  * Test BlobQuery with wildcard patterns
  */
 TEST_CASE("BlobQuery - Wildcard Patterns", "[query][blobquery][wildcard]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing BlobQuery with wildcard patterns");

   // Query for all .dat blobs in user_data tag
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::BlobQueryAsync(cte_client, "user_data", "blob_.*\\.dat", 0, chi::PoolQuery::Broadcast());

   INFO("Total blobs matched: " << results.size());
   REQUIRE(results.size() >= 2); // Should match blob_001.dat and blob_002.dat

   int dat_blob_count = 0;
   for (const auto &pair : results) {
     INFO("Found blob: " << pair.first << "/" << pair.second);
     if (pair.first == "user_data" &&
         pair.second.find("blob_") != std::string::npos &&
         pair.second.find(".dat") != std::string::npos) {
       dat_blob_count++;
     }
   }
   REQUIRE(dat_blob_count >= 2);
 }

 /**
  * Test BlobQuery with multiple tag matches
  */
 TEST_CASE("BlobQuery - Multiple Tags", "[query][blobquery][multitag]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing BlobQuery with multiple tag matches");

   // Query for all .txt files in any "user_" tag
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::BlobQueryAsync(cte_client, "user_.*", "file_.*\\.txt", 0, chi::PoolQuery::Broadcast());

   INFO("Total blobs matched: " << results.size());
   REQUIRE(results.size() >= 4); // user_data and user_logs each have 2 .txt files

   int txt_file_count = 0;
   for (const auto &pair : results) {
     INFO("Found blob: " << pair.first << "/" << pair.second);
     if (pair.second.find("file_") != std::string::npos && pair.second.find(".txt") != std::string::npos) {
       txt_file_count++;
     }
   }
   REQUIRE(txt_file_count >= 4);
 }

 /**
  * Test BlobQuery with match-all patterns
  */
 TEST_CASE("BlobQuery - Match All", "[query][blobquery][matchall]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing BlobQuery with match-all patterns");

   // Query for all blobs in all tags
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::BlobQueryAsync(cte_client, ".*", ".*", 0, chi::PoolQuery::Broadcast());

   INFO("Total blobs matched: " << results.size());
   REQUIRE(results.size() >= fixture->test_blobs_.size());
 }

 /**
  * Test BlobQuery with no blob matches
  */
 TEST_CASE("BlobQuery - No Blob Matches", "[query][blobquery][noblob]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing BlobQuery with blob pattern that matches nothing");

   // Query for non-existent blob pattern in existing tag
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::BlobQueryAsync(cte_client, "user_data", "nonexistent_blob_xyz", 0, chi::PoolQuery::Broadcast());

   INFO("Total blobs matched: " << results.size());
   REQUIRE(results.size() == 0);
 }

 /**
  * Test BlobQuery with no tag matches
  */
 TEST_CASE("BlobQuery - No Tag Matches", "[query][blobquery][notag]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing BlobQuery with tag pattern that matches nothing");

   // Query for non-existent tag pattern
   auto *cte_client = WRP_CTE_CLIENT;
   auto results = CTEQueryTestFixture::BlobQueryAsync(cte_client, "nonexistent_tag_xyz", ".*", 0, chi::PoolQuery::Broadcast());

   INFO("Query returned " << results.size() << " blob pairs");
   REQUIRE(results.empty());
 }

 /**
  * Test Query API with Local pool query
  */
 TEST_CASE("Query - Local Pool Query", "[query][poolquery][local]") {
   auto *fixture = hshm::Singleton<CTEQueryTestFixture>::GetInstance();
  (void)fixture; // Suppress unused variable warning
   INFO("Testing query APIs with Local pool query");

   auto *cte_client = WRP_CTE_CLIENT;

   // TagQuery with Local should work but only return local results
   auto tag_results = CTEQueryTestFixture::TagQueryAsync(cte_client, "user_.*", 0, chi::PoolQuery::Local());

   INFO("TagQuery with Local returned " << tag_results.size() << " tags");
   // Should get results since tags were created locally
   REQUIRE(!tag_results.empty());

   // BlobQuery with Local should also work
   auto blob_results = CTEQueryTestFixture::BlobQueryAsync(cte_client, "user_.*", "blob_.*", 0, chi::PoolQuery::Local());

   INFO("BlobQuery with Local returned " << blob_results.size() << " blob pairs");
   REQUIRE(blob_results.size() > 0);
 }

// Main function using simple_test.h framework
SIMPLE_TEST_MAIN()
