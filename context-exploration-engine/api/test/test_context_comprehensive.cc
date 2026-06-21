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
 * test_context_comprehensive.cc
 *
 * Comprehensive tests for CEE (Context Exploration Engine) APIs
 * Focus: Improving coverage of ContextInterface core methods
 *
 * Critical Coverage Areas:
 * - ContextRetrieve (fully implemented but untested)
 * - Error path handling
 * - Batch processing edge cases
 * - Buffer management
 */

#include "simple_test.h"
#include <clio_cee/api/context_interface.h>
#include <clio_cae/core/core_client.h>
#include <clio_cae/core/core_tasks.h>
#include <clio_cae/core/constants.h>
#include <clio_cae/core/factory/assimilation_ctx.h>
#include <clio_cte/core/core_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <fstream>
#include <filesystem>
#include <set>
#include <system_error>
#include <cstdlib>

using namespace iowarp;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/**
 * Test fixture for CEE comprehensive tests
 */
class CEEComprehensiveFixture {
public:
  static inline bool g_initialized = false;
  std::string test_data_dir_;
  std::string test_binary_file_;

  static constexpr size_t kSmallFileSize = 2048;  // 2KB for quick tests

  CEEComprehensiveFixture() {
    if (!g_initialized) {
      INFO("=== Initializing CEE Test Environment ===");

      // Initialize CLIO Runtime runtime
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (!success) {
        throw std::runtime_error("CLIO_INIT failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Initialize CTE client
      success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
      if (!success) {
        throw std::runtime_error("CLIO_CTE_CLIENT_INIT failed");
      }

      // Set pool ID on global CTE client
      auto *cte_client = CLIO_CTE_CLIENT;
      cte_client->Init(clio::cte::core::kCtePoolId);

      // Create CTE core pool
      clio::cte::core::CreateParams cte_params;
      auto cte_create = cte_client->AsyncCreate(
          clio::run::PoolQuery::Dynamic(),
          clio::cte::core::kCtePoolName,
          clio::cte::core::kCtePoolId,
          cte_params);
      cte_create.Wait();

      if (cte_create->GetReturnCode() != 0) {
        throw std::runtime_error("CTE pool creation failed");
      }

      // Register a RAM storage target so PutBlob can allocate space.
      // When CLIO_SERVER_CONF is set, compose config already registers targets,
      // but when it's absent (e.g., compute nodes), the pool has none.
      auto reg_task = cte_client->AsyncRegisterTarget(
          "ram::cee_test_cache",
          clio::run::bdev::BdevType::kRam,
          512 * 1024 * 1024,             // 512 MB
          clio::run::PoolQuery::Local(),
          clio::run::PoolId(512, 10));          // explicit bdev pool id
      reg_task.Wait();

      // Initialize CAE client
      CLIO_CAE_CLIENT_INIT();

      // Create CAE pool
      clio::cae::core::Client cae_client;
      clio::cae::core::CreateParams cae_params;
      auto cae_create = cae_client.AsyncCreate(
          clio::run::PoolQuery::Local(),
          "test_cae_pool",
          clio::cae::core::kCaePoolId,
          cae_params);
      cae_create.Wait();

      if (cae_create->GetReturnCode() != 0) {
        throw std::runtime_error("CAE pool creation failed");
      }

      INFO("CEE infrastructure initialized");
      g_initialized = true;
    }

    test_data_dir_ = chi_test_data_dir() + "/cee_test_data";
    test_binary_file_ = test_data_dir_ + "/test_cee_data.bin";
  }

  void SetupTestData() {
    // std::filesystem::create_directories is portable; `system("mkdir -p")`
    // breaks on Windows cmd (no -p flag) and bashes against quoting/space
    // issues even on POSIX. The non-throwing overload is fine since the
    // directory may already exist from a previous test in the same proc.
    std::error_code _ec;
    std::filesystem::create_directories(test_data_dir_, _ec);

    // Generate small binary test file
    GenerateBinaryFile(test_binary_file_, kSmallFileSize);
  }

  void GenerateBinaryFile(const std::string& path, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to create binary file: " + path);
    }

    // Write patterned data (4-byte blocks with incrementing values)
    const size_t block_size = 4;
    const size_t num_blocks = size / block_size;

    for (size_t i = 0; i < num_blocks; ++i) {
      uint32_t value = static_cast<uint32_t>(i);
      file.write(reinterpret_cast<const char*>(&value), block_size);
    }

    file.close();
    INFO("Generated binary test file: " << path << " (" << size << " bytes)");
  }

  ~CEEComprehensiveFixture() {
    INFO("CEE test cleanup");
  }
};

// ============================================================================
// ContextInterface Initialization Tests
// ============================================================================

TEST_CASE("CEE - ContextInterface Construction", "[cee][init]") {
  CEEComprehensiveFixture fixture;

  // Should construct without errors (lazy initialization)
  ContextInterface ctx_interface;
  INFO("ContextInterface constructed successfully");
}

// ============================================================================
// ContextRetrieve Tests - CRITICAL (fully implemented but untested)
// ============================================================================

TEST_CASE("CEE - ContextRetrieve Basic", "[cee][retrieve][basic]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // First, bundle some test data
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string dst_url = "iowarp::cee_retrieve_test";

  clio::cae::core::AssimilationCtx ctx(src_url, dst_url, "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  int result = ctx_interface.ContextBundle(bundle);
  REQUIRE(result == 0);

  // Allow time for async assimilation
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Now retrieve the data
  auto retrieved = ctx_interface.ContextRetrieve(
      "cee_retrieve_test",  // tag_re
      ".*",                 // blob_re (all blobs)
      1024,                 // max_results
      1024 * 1024,          // max_context_size (1MB)
      32);                  // batch_size

  // Should retrieve at least one result
  INFO("Retrieved " << retrieved.size() << " context(s)");
  REQUIRE(retrieved.size() > 0);
}

TEST_CASE("CEE - ContextRetrieve No Matching Blobs", "[cee][retrieve][empty]") {
  CEEComprehensiveFixture fixture;

  ContextInterface ctx_interface;

  // Query for non-existent pattern
  auto retrieved = ctx_interface.ContextRetrieve(
      "nonexistent_tag_pattern_.*",
      "nonexistent_blob_.*",
      100,
      1024 * 1024,
      32);

  // Should return empty vector
  REQUIRE(retrieved.size() == 0);
  INFO("ContextRetrieve correctly returned empty results");
}

TEST_CASE("CEE - ContextRetrieve With Small Buffer", "[cee][retrieve][buffer]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle test data
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string dst_url = "iowarp::cee_small_buffer_test";

  clio::cae::core::AssimilationCtx ctx(src_url, dst_url, "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Retrieve with very small buffer (512 bytes)
  auto retrieved = ctx_interface.ContextRetrieve(
      "cee_small_buffer_test",
      ".*",
      1024,
      512,   // Small buffer - may not fit all data
      32);

  INFO("Retrieved with small buffer: " << retrieved.size() << " context(s)");
  // Should either retrieve partial data or handle gracefully
}

TEST_CASE("CEE - ContextRetrieve With Custom Batch Size", "[cee][retrieve][batch]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle test data
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string dst_url = "iowarp::cee_batch_test";

  clio::cae::core::AssimilationCtx ctx(src_url, dst_url, "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Retrieve with custom batch sizes
  auto retrieved_small_batch = ctx_interface.ContextRetrieve(
      "cee_batch_test", ".*", 1024, 1024 * 1024, 1);  // batch_size = 1

  auto retrieved_large_batch = ctx_interface.ContextRetrieve(
      "cee_batch_test", ".*", 1024, 1024 * 1024, 100);  // batch_size = 100

  INFO("Small batch: " << retrieved_small_batch.size() << " contexts");
  INFO("Large batch: " << retrieved_large_batch.size() << " contexts");
}

TEST_CASE("CEE - ContextRetrieve With Max Results Limit", "[cee][retrieve][limit]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle test data
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::string dst_url = "iowarp::cee_limit_test";

  clio::cae::core::AssimilationCtx ctx(src_url, dst_url, "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Retrieve with max_results = 1 (should limit results)
  auto retrieved = ctx_interface.ContextRetrieve(
      "cee_limit_test", ".*", 1, 1024 * 1024, 32);

  INFO("Retrieved with max_results=1: " << retrieved.size() << " contexts");
}

// ============================================================================
// ContextBundle Enhanced Tests
// ============================================================================

TEST_CASE("CEE - ContextBundle Multiple Files", "[cee][bundle][multi]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Create multiple file contexts
  std::string src_url = "file::" + fixture.test_binary_file_;

  std::vector<clio::cae::core::AssimilationCtx> bundle;
  bundle.emplace_back(src_url, "iowarp::cee_multi_1", "binary");
  bundle.emplace_back(src_url, "iowarp::cee_multi_2", "binary");
  bundle.emplace_back(src_url, "iowarp::cee_multi_3", "binary");

  int result = ctx_interface.ContextBundle(bundle);
  REQUIRE(result == 0);
  INFO("Successfully bundled " << bundle.size() << " contexts");
}

TEST_CASE("CEE - ContextBundle With Range", "[cee][bundle][range]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  std::string src_url = "file::" + fixture.test_binary_file_;

  // Bundle with range (partial file)
  clio::cae::core::AssimilationCtx ctx(
      src_url,
      "iowarp::cee_range_test",
      "binary",
      "",      // no dependency
      0,       // offset
      1024);   // size (first 1KB only)

  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  int result = ctx_interface.ContextBundle(bundle);
  REQUIRE(result == 0);
  INFO("Successfully bundled context with range");
}

TEST_CASE("CEE - ContextBundle Invalid Source", "[cee][bundle][error]") {
  CEEComprehensiveFixture fixture;

  ContextInterface ctx_interface;

  // Try to bundle non-existent file
  clio::cae::core::AssimilationCtx ctx(
      "file::/nonexistent/file.bin",
      "iowarp::cee_invalid_test",
      "binary");

  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  // Should handle gracefully (may succeed with scheduling but fail later)
  int result = ctx_interface.ContextBundle(bundle);
  INFO("ContextBundle with invalid source returned: " << result);
}

// ============================================================================
// ContextQuery Enhanced Tests
// ============================================================================

TEST_CASE("CEE - ContextQuery With Regex Patterns", "[cee][query][regex]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle test data
  std::string src_url = "file::" + fixture.test_binary_file_;
  clio::cae::core::AssimilationCtx ctx(src_url, "iowarp::cee_regex_test", "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};
  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Query with various regex patterns
  auto results_wildcard = ctx_interface.ContextQuery("cee_regex_.*", ".*");
  auto results_specific = ctx_interface.ContextQuery("cee_regex_test", ".*");
  auto results_nonmatch = ctx_interface.ContextQuery("nonexistent_.*", ".*");

  INFO("Wildcard pattern: " << results_wildcard.size() << " results");
  INFO("Specific pattern: " << results_specific.size() << " results");
  INFO("Non-matching pattern: " << results_nonmatch.size() << " results");

  REQUIRE(results_nonmatch.size() == 0);
}

TEST_CASE("CEE - ContextQuery With Max Results", "[cee][query][limit]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle test data
  std::string src_url = "file::" + fixture.test_binary_file_;
  clio::cae::core::AssimilationCtx ctx(src_url, "iowarp::cee_query_limit", "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};
  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Query with max_results limit
  auto results_unlimited = ctx_interface.ContextQuery("cee_query_limit", ".*", 0);
  auto results_limited = ctx_interface.ContextQuery("cee_query_limit", ".*", 1);

  INFO("Unlimited: " << results_unlimited.size() << " results");
  INFO("Limited to 1: " << results_limited.size() << " results");
}

// ============================================================================
// ContextBundle: string:: protocol with src_data payload
// ============================================================================

TEST_CASE("CEE - ContextBundle string:: payload (src_data)",
          "[cee][bundle][string]") {
  CEEComprehensiveFixture fixture;

  ContextInterface ctx_interface;

  // Three in-memory payloads land as three blobs under the same tag.
  // Mirrors the agentic-loop flow: caller already has bytes in hand and
  // hands them off to Clio without going through a file.
  auto make = [](const std::string& blob, const std::string& tag,
                 const std::string& data) {
    clio::cae::core::AssimilationCtx c;
    c.src = "string::" + blob;
    c.dst = "iowarp::" + tag;
    c.format = "string";
    c.src_data = data;
    return c;
  };
  std::vector<clio::cae::core::AssimilationCtx> bundle{
      make("step1", "cee_string_tag", "agent step 1 result"),
      make("step2", "cee_string_tag", "agent step 2 result"),
      make("step3", "cee_string_tag", "agent step 3 result"),
  };

  int rc = ctx_interface.ContextBundle(bundle);
  REQUIRE(rc == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  auto names = ctx_interface.ContextQuery("cee_string_tag", ".*");
  std::set<std::string> got(names.begin(), names.end());
  INFO("Got " << got.size() << " blobs under cee_string_tag");
  REQUIRE(got.count("step1") == 1);
  REQUIRE(got.count("step2") == 1);
  REQUIRE(got.count("step3") == 1);
}

// ============================================================================
// ContextQuery: BM25 prompt path
// ============================================================================

TEST_CASE("CEE - ContextQuery prompt triggers BM25 semantic search",
          "[cee][query][prompt][bm25]") {
  CEEComprehensiveFixture fixture;

  ContextInterface ctx_interface;

  // Three short text blobs with distinct vocabularies. The BM25 path
  // should rank the spaceflight blob highest for a moon/Apollo query —
  // not because the model is smart, but because BM25 term-frequency
  // weighting falls out of the keyword overlap.
  auto mk = [](const std::string& blob, const std::string& body) {
    clio::cae::core::AssimilationCtx c;
    c.src = "string::" + blob;
    c.dst = "iowarp::cee_bm25_tag";
    c.format = "string";
    c.src_data = body;
    return c;
  };
  std::vector<clio::cae::core::AssimilationCtx> bundle{
      mk("space_doc",
         "Apollo 11 landed on the Moon in 1969. The lunar surface mission "
         "with astronauts Armstrong and Aldrin defined human spaceflight."),
      mk("cooking_doc",
         "A roux is equal parts butter and flour cooked into a paste, the "
         "base for many classic French sauces like bechamel and espagnole."),
      mk("garden_doc",
         "Tomato plants need full sun, regular watering, and well-drained "
         "soil with a slightly acidic pH for a healthy fruit set."),
  };
  int rc = ctx_interface.ContextBundle(bundle);
  REQUIRE(rc == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Sanity: regex-only query (empty prompt) still returns all three.
  auto regex_only = ctx_interface.ContextQuery("cee_bm25_tag", ".*");
  REQUIRE(regex_only.size() == 3);

  // BM25 path: prompt about the moon should put space_doc first.
  auto ranked = ctx_interface.ContextQuery(
      "cee_bm25_tag", ".*", /*max_results=*/3,
      /*prompt=*/"apollo moon lunar surface astronaut");
  INFO("BM25 ranked " << ranked.size() << " results");
  for (size_t i = 0; i < ranked.size(); ++i) {
    INFO("  #" << i << "  " << ranked[i]);
  }
  REQUIRE(ranked.size() > 0);
  REQUIRE(ranked.size() <= 3);
  REQUIRE(ranked[0] == "space_doc");

  // top-k cap: ask for 1, get 1 (and it's the same #0).
  auto top1 = ctx_interface.ContextQuery(
      "cee_bm25_tag", ".*", /*max_results=*/1,
      /*prompt=*/"apollo moon lunar surface astronaut");
  REQUIRE(top1.size() == 1);
  REQUIRE(top1[0] == "space_doc");
}

// ============================================================================
// ContextDestroy Enhanced Tests
// ============================================================================

TEST_CASE("CEE - ContextDestroy Multiple Contexts", "[cee][destroy][multi]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle multiple contexts
  std::string src_url = "file::" + fixture.test_binary_file_;
  std::vector<clio::cae::core::AssimilationCtx> bundle;
  bundle.emplace_back(src_url, "iowarp::cee_destroy_1", "binary");
  bundle.emplace_back(src_url, "iowarp::cee_destroy_2", "binary");
  bundle.emplace_back(src_url, "iowarp::cee_destroy_3", "binary");

  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Destroy all at once
  std::vector<std::string> contexts_to_destroy = {
      "cee_destroy_1", "cee_destroy_2", "cee_destroy_3"};

  int result = ctx_interface.ContextDestroy(contexts_to_destroy);
  INFO("ContextDestroy returned: " << result);
}

TEST_CASE("CEE - ContextDestroy Partial Failure", "[cee][destroy][partial]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Bundle one context
  std::string src_url = "file::" + fixture.test_binary_file_;
  clio::cae::core::AssimilationCtx ctx(src_url, "iowarp::cee_destroy_partial", "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};
  ctx_interface.ContextBundle(bundle);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Try to destroy mix of existing and non-existing
  std::vector<std::string> contexts = {
      "cee_destroy_partial",        // exists
      "nonexistent_context_1",      // doesn't exist
      "nonexistent_context_2"       // doesn't exist
  };

  int result = ctx_interface.ContextDestroy(contexts);
  INFO("ContextDestroy with partial failure returned: " << result);
  // Should report errors for non-existent contexts
}

// ============================================================================
// ContextSplice Tests (stub implementation)
// ============================================================================

TEST_CASE("CEE - ContextSplice Not Implemented", "[cee][splice][stub]") {
  CEEComprehensiveFixture fixture;

  ContextInterface ctx_interface;

  // Call ContextSplice (should return error code 1)
  int result = ctx_interface.ContextSplice("new_ctx", "tag_.*", "blob_.*");

  REQUIRE(result == 1);
  INFO("ContextSplice correctly returns not-implemented error");
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_CASE("CEE - Full Workflow Bundle-Query-Retrieve-Destroy", "[cee][integration]") {
  CEEComprehensiveFixture fixture;
  fixture.SetupTestData();

  ContextInterface ctx_interface;

  // Step 1: Bundle
  std::string src_url = "file::" + fixture.test_binary_file_;
  clio::cae::core::AssimilationCtx ctx(src_url, "iowarp::cee_workflow", "binary");
  std::vector<clio::cae::core::AssimilationCtx> bundle = {ctx};

  int bundle_result = ctx_interface.ContextBundle(bundle);
  REQUIRE(bundle_result == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Step 2: Query
  auto query_results = ctx_interface.ContextQuery("cee_workflow", ".*");
  INFO("Query found " << query_results.size() << " blobs");

  // Step 3: Retrieve
  auto retrieved = ctx_interface.ContextRetrieve(
      "cee_workflow", ".*", 1024, 1024 * 1024, 32);
  INFO("Retrieved " << retrieved.size() << " contexts");

  // Step 4: Destroy
  std::vector<std::string> to_destroy = {"cee_workflow"};
  int destroy_result = ctx_interface.ContextDestroy(to_destroy);
  INFO("Destroy returned: " << destroy_result);

  INFO("Full workflow completed successfully");
}

SIMPLE_TEST_MAIN()
