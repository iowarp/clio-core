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
 * EVICT TEST
 *
 * Exercises Client::AsyncEvict (Method::kEvict): score-driven capacity
 * reclamation. Given a "minimum tier score" and a byte budget, the runtime
 * frees the LOWEST-score blobs occupying qualifying tiers, cheapest-first,
 * until the budget is met.
 *
 * Config: a single DRAM tier (score 1.0) so placement is deterministic and a
 * blob's own put-score becomes its rank key. Four 1 MB blobs are stored with
 * distinct scores; Evict must remove the lowest-scoring ones first.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;  // 1MB per blob
static constexpr float kTierScore = 1.0f;                     // single DRAM tier

class EvictTestFixture {
 public:
  std::string config_path_;
  bool initialized_ = false;

  EvictTestFixture() {
    config_path_ = chi_test_data_dir() + "/evict_config.yaml";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    initialized_ = true;
  }

  ~EvictTestFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Evict Test Configuration - single 64MB DRAM tier (score 1.0)
runtime:
  num_threads: 2
  queue_depth: 1024
  first_busy_wait: 10000
  max_sleep: 50000

compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    storage:
      - path: "ram::evict_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }

  // Store one 1MB blob with the given put-score (which becomes its rank key).
  void PutScoredBlob(clio::cte::core::Tag &tag, const std::string &blob_name,
                     float score, char pattern) {
    auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE(!shm_buffer.IsNull());
    std::memset(shm_buffer.ptr_, pattern, kBlobSize);
    auto put_task = tag.AsyncPutBlob(
        blob_name, shm_buffer.shm_.template Cast<void>(), kBlobSize, 0, score);
    put_task.Wait();
    REQUIRE(put_task->GetReturnCode() == 0);
    CLIO_IPC->FreeBuffer(shm_buffer);
  }
};

static EvictTestFixture *g_fixture = nullptr;

// Returns true if the blob still exists (GetBlobScore succeeds).
static bool BlobExists(clio::cte::core::Client *cte_client,
                       const clio::cte::core::TagId &tag_id,
                       const std::string &blob_name) {
  auto score_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_task.Wait();
  return score_task->GetReturnCode() == 0;
}

/**
 * Evict the two lowest-score blobs off the tier via a 2 MB budget.
 */
TEST_CASE("Evict - lowest score first until budget met", "[evict][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "evict_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  // Four 1MB blobs with distinct scores on the single DRAM tier.
  g_fixture->PutScoredBlob(tag, "blob_hi", 1.0f, 'A');    // keep
  g_fixture->PutScoredBlob(tag, "blob_mid", 0.75f, 'B');  // keep
  g_fixture->PutScoredBlob(tag, "blob_lo", 0.50f, 'C');   // evict (2nd)
  g_fixture->PutScoredBlob(tag, "blob_lowest", 0.25f, 'D');  // evict (1st)

  REQUIRE(BlobExists(cte_client, tag_id, "blob_lowest"));
  REQUIRE(BlobExists(cte_client, tag_id, "blob_hi"));

  // Evict 2MB from any tier with score >= 0.5 (the DRAM tier at 1.0 qualifies).
  const clio::run::u64 kBudget = 2 * kBlobSize;
  auto evict_task = cte_client->AsyncEvict(/*min_tier_score=*/0.5f, kBudget);
  evict_task.Wait();
  REQUIRE(evict_task->GetReturnCode() == 0);

  INFO("bytes_evicted=" << evict_task->bytes_evicted_
                        << " blobs_evicted=" << evict_task->blobs_evicted_);

  // Budget satisfied by exactly the two lowest-score blobs.
  REQUIRE(evict_task->bytes_evicted_ >= kBudget);
  REQUIRE(evict_task->blobs_evicted_ == 2);

  // The two lowest-score blobs are gone; the two highest remain.
  REQUIRE(!BlobExists(cte_client, tag_id, "blob_lowest"));
  REQUIRE(!BlobExists(cte_client, tag_id, "blob_lo"));
  REQUIRE(BlobExists(cte_client, tag_id, "blob_mid"));
  REQUIRE(BlobExists(cte_client, tag_id, "blob_hi"));

  // Clean up the survivors.
  cte_client->AsyncDelBlob(tag_id, "blob_mid").Wait();
  cte_client->AsyncDelBlob(tag_id, "blob_hi").Wait();
}

/**
 * A min_tier_score above every registered target evicts nothing.
 */
TEST_CASE("Evict - no qualifying tier is a no-op", "[evict][noleak]") {
  REQUIRE(g_fixture != nullptr);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "evict_noop_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  g_fixture->PutScoredBlob(tag, "keep_me", 0.5f, 'K');
  REQUIRE(BlobExists(cte_client, tag_id, "keep_me"));

  // No target has score >= 2.0, so nothing should be evicted.
  auto evict_task = cte_client->AsyncEvict(/*min_tier_score=*/2.0f, kBlobSize);
  evict_task.Wait();
  REQUIRE(evict_task->GetReturnCode() == 0);
  REQUIRE(evict_task->bytes_evicted_ == 0);
  REQUIRE(evict_task->blobs_evicted_ == 0);
  REQUIRE(BlobExists(cte_client, tag_id, "keep_me"));

  cte_client->AsyncDelBlob(tag_id, "keep_me").Wait();
}

int main(int argc, char **argv) {
  g_fixture = new EvictTestFixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;
}
