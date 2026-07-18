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
 * DATA ORGANIZER TEST (issue #738)
 *
 * Exercises the internal, periodically-driven data organizer:
 * - DataOrganizerFactory name resolution
 * - FrecencyDataOrganizer::ComputeScore (pure scoring function)
 * - Organizer config parsing (organizer / organizer_tasks /
 *   organizer_period_ms)
 * - End-to-end: the periodic DynamicReorganize task rescores blobs by
 *   frecency — a frequently-read blob placed cold floats up, an untouched
 *   blob placed hot sinks down — with data integrity preserved across the
 *   internally-driven moves.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/data_organizer/data_organizer.h>
#include <clio_cte/core/data_organizer/frecency_organizer.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

// Two-tier storage: fast DRAM (score 1.0) over a slow file tier (score 0.2)
static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;  // 1MB per blob

// Fast organizer cadence so the test observes rescoring within seconds
static constexpr clio::run::u32 kOrganizerPeriodMs = 500;

/**
 * Test fixture: two-tier CTE with the frecency organizer enabled
 * (organizer_tasks: 2 exercises the replica partitioning).
 */
class DataOrganizerTestFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  DataOrganizerTestFixture() {
    INFO("=== Initializing DataOrganizer Test ===");

    config_path_ = chi_test_data_dir() + "/data_organizer_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/data_organizer_storage.bin";

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
    INFO("=== DataOrganizer Test Environment Ready ===");
  }

  ~DataOrganizerTestFixture() {
    INFO("=== Cleaning up DataOrganizer Test ===");
    Cleanup();
  }

  void Cleanup() {
    if (fs::exists(config_path_)) {
      fs::remove(config_path_);
    }
    if (fs::exists(file_storage_path_)) {
      fs::remove(file_storage_path_);
    }
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# DataOrganizer Test Configuration
# - 16MB DRAM (fast tier, score 1.0)
# - 64MB File (slow tier, score 0.2)
# - frecency organizer, 2 replicas, )" << kOrganizerPeriodMs << R"( ms period

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
      # Fast tier: 16MB DRAM (score 1.0)
      - path: "ram::organizer_dram"
        bdev_type: "ram"
        capacity_limit: "16MB"
        score: 1.0

      # Slow tier: 64MB File (score 0.2)
      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 0.2

    dpe:
      dpe_type: "max_bw"

    organizer: "frecency"
    organizer_tasks: 2
    organizer_period_ms: )" << kOrganizerPeriodMs << R"(
)";

    config_file.close();
    INFO("Created config file: " << config_path_);
  }

  std::vector<char> CreateTestData(size_t size, char pattern) {
    std::vector<char> data(size);
    std::memset(data.data(), pattern, size);
    return data;
  }

  bool VerifyTestData(const std::vector<char> &data, char expected_pattern) {
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] != expected_pattern) {
        return false;
      }
    }
    return true;
  }
};

// Global fixture instance
static DataOrganizerTestFixture *g_fixture = nullptr;

/**
 * Test: factory resolves organizer names
 */
TEST_CASE("DataOrganizer - Factory", "[organizer][factory]") {
  using clio::cte::core::DataOrganizerFactory;

  auto frecency = DataOrganizerFactory::Get("frecency");
  REQUIRE(frecency != nullptr);
  REQUIRE(frecency->GetName() == "frecency");

  REQUIRE(DataOrganizerFactory::Get("none") == nullptr);
  REQUIRE(DataOrganizerFactory::Get("") == nullptr);
  REQUIRE(DataOrganizerFactory::Get("no_such_organizer") == nullptr);

  INFO("SUCCESS: factory name resolution");
}

/**
 * Test: frecency scoring math — hot beats cold, bounds hold
 */
TEST_CASE("DataOrganizer - Frecency Score", "[organizer][frecency][score]") {
  using clio::cte::core::FrecencyDataOrganizer;
  using clio::cte::core::OrganizerBlobStat;
  using clio::cte::core::Timestamp;

  // Arbitrary steady-clock ns — large enough that subtracting ten recency
  // half-lives (6e12 ns) below cannot underflow the unsigned timestamp.
  const Timestamp now = 10'000'000'000'000'000ULL;

  // Hot: accessed right now, many times
  OrganizerBlobStat hot;
  hot.last_read_ = now;
  hot.access_count_ = 100;
  float hot_score = FrecencyDataOrganizer::ComputeScore(hot, now);

  // Cold: accessed once, ten half-lives ago
  OrganizerBlobStat cold;
  cold.last_read_ = now -
      static_cast<Timestamp>(10.0 * FrecencyDataOrganizer::kRecencyHalfLifeSec *
                             1e9);
  cold.access_count_ = 1;
  float cold_score = FrecencyDataOrganizer::ComputeScore(cold, now);

  INFO("hot_score=" << hot_score << " cold_score=" << cold_score);
  REQUIRE(hot_score > cold_score);
  REQUIRE(hot_score > 0.8f);
  REQUIRE(cold_score < 0.2f);
  REQUIRE(hot_score <= 1.0f);
  REQUIRE(cold_score >= 0.0f);

  // Recency alone (single access, just now) lands mid-scale
  OrganizerBlobStat fresh;
  fresh.last_modified_ = now;
  fresh.access_count_ = 1;
  float fresh_score = FrecencyDataOrganizer::ComputeScore(fresh, now);
  REQUIRE(fresh_score > 0.4f);
  REQUIRE(fresh_score < 0.7f);

  // Timestamps after `now` (clock skew between snapshot and stamp) must not
  // blow up — treated as age 0.
  OrganizerBlobStat skewed;
  skewed.last_read_ = now + 1'000'000ULL;
  skewed.access_count_ = 1;
  float skewed_score = FrecencyDataOrganizer::ComputeScore(skewed, now);
  REQUIRE(skewed_score >= 0.0f);
  REQUIRE(skewed_score <= 1.0f);

  INFO("SUCCESS: frecency scoring math");
}

/**
 * Test: organizer config keys parse (and invalid names are rejected)
 */
TEST_CASE("DataOrganizer - Config Parse", "[organizer][config]") {
  clio::cte::core::Config config;

  bool ok = config.LoadFromString(R"(
organizer: "frecency"
organizer_tasks: 4
organizer_period_ms: 250
)");
  REQUIRE(ok);
  REQUIRE(config.organizer_.name_ == "frecency");
  REQUIRE(config.organizer_.organizer_tasks_ == 4);
  REQUIRE(config.organizer_.period_ms_ == 250);

  // Defaults: disabled, 1 task
  clio::cte::core::Config defaults;
  REQUIRE(defaults.organizer_.name_ == "none");
  REQUIRE(defaults.organizer_.organizer_tasks_ == 1);

  // Invalid organizer name is rejected
  clio::cte::core::Config bad;
  REQUIRE(!bad.LoadFromString("organizer: \"bogus\"\n"));

  // organizer_tasks: 0 is rejected
  clio::cte::core::Config zero_tasks;
  REQUIRE(!zero_tasks.LoadFromString("organizer_tasks: 0\n"));

  INFO("SUCCESS: organizer config parsing");
}

/**
 * Test: a frequently-read blob placed on the cold tier floats up
 *
 * The blob starts at score 0.2 and is then read repeatedly. Frecency for a
 * just-read blob with ~30 accesses is ~0.85 (recency ~1.0, frequency
 * ~30/40), so after a few DynamicReorganize periods the score must have
 * risen well past 0.5 without any explicit ReorganizeBlob call.
 */
TEST_CASE("DataOrganizer - Hot Blob Promoted", "[organizer][promote][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("organizer_test_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "hot_blob";

  // Put on the cold tier (score 0.2)
  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  auto test_data = g_fixture->CreateTestData(kBlobSize, 'H');
  std::memcpy(shm_buffer.ptr_, test_data.data(), kBlobSize);

  auto put_task = tag.AsyncPutBlob(blob_name,
                                   shm_buffer.shm_.template Cast<void>(),
                                   kBlobSize, 0, 0.2f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  // Read the blob repeatedly to build up frequency + recency. On a slow
  // (Debug/CI) machine this loop overlaps organizer rounds, and the current
  // reorganize implementation deletes the blob's metadata for the duration
  // of a move — a concurrent read then transiently fails with "not found"
  // (issue #753). Tolerate those: a failed read simply doesn't count as an
  // access. Enough must succeed to build the frequency signal.
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());
  int ok_reads = 0;
  for (int i = 0; i < 30; ++i) {
    auto get_task = cte_client->AsyncGetBlob(
        tag_id, blob_name, 0, kBlobSize, 0,
        read_buffer.shm_.template Cast<void>());
    get_task.Wait();
    if (get_task->GetReturnCode() == 0) {
      ok_reads++;
    }
  }
  INFO("successful reads: " << ok_reads << "/30");
  REQUIRE(ok_reads >= 10);

  // Let several organizer periods elapse
  std::this_thread::sleep_for(
      std::chrono::milliseconds(5 * kOrganizerPeriodMs));

  float score = -1.0f;
  for (int attempt = 0; attempt < 10; ++attempt) {
    auto score_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
    score_task.Wait();
    if (score_task->GetReturnCode() == 0) {
      score = score_task->score_;
      break;
    }
    // Mid-move window (issue #753) — retry.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  INFO("hot blob score after organizer rounds: " << score);
  REQUIRE(score > 0.5f);

  // Data must have survived the internally-driven move. Retry briefly in
  // case a read lands inside a move window (issue #753).
  bool data_valid = false;
  for (int attempt = 0; attempt < 10 && !data_valid; ++attempt) {
    std::memset(read_buffer.ptr_, 0, kBlobSize);
    auto get_task = cte_client->AsyncGetBlob(
        tag_id, blob_name, 0, kBlobSize, 0,
        read_buffer.shm_.template Cast<void>());
    get_task.Wait();
    if (get_task->GetReturnCode() != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    std::vector<char> read_data(kBlobSize);
    std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);
    data_valid = g_fixture->VerifyTestData(read_data, 'H');
  }
  REQUIRE(data_valid);

  CLIO_IPC->FreeBuffer(shm_buffer);
  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: hot blob promoted by the frecency organizer");
}

/**
 * Test: an untouched blob placed on the fast tier sinks toward its frecency
 *
 * The blob starts at score 1.0 and is never accessed again. Its frecency is
 * ~0.55 (recency ~1.0 within the 10-min half-life, frequency 1/11), so
 * after a few periods the organizer must have pulled the score down below
 * 0.7 without any explicit ReorganizeBlob call.
 */
TEST_CASE("DataOrganizer - Cold Blob Demoted", "[organizer][demote][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("organizer_test_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "cold_blob";

  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  auto test_data = g_fixture->CreateTestData(kBlobSize, 'C');
  std::memcpy(shm_buffer.ptr_, test_data.data(), kBlobSize);

  auto put_task = tag.AsyncPutBlob(blob_name,
                                   shm_buffer.shm_.template Cast<void>(),
                                   kBlobSize, 0, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  // No further accesses; let several organizer periods elapse
  std::this_thread::sleep_for(
      std::chrono::milliseconds(5 * kOrganizerPeriodMs));

  float score = -1.0f;
  for (int attempt = 0; attempt < 10; ++attempt) {
    auto score_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
    score_task.Wait();
    if (score_task->GetReturnCode() == 0) {
      score = score_task->score_;
      break;
    }
    // Mid-move window (issue #753) — retry.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  INFO("cold blob score after organizer rounds: " << score);
  REQUIRE(score < 0.7f);
  REQUIRE(score > 0.2f);  // still recent — not bottomed out

  // Data integrity across the internally-driven demotion. Retry briefly in
  // case the read lands inside a move window (issue #753).
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());
  bool data_valid = false;
  for (int attempt = 0; attempt < 10 && !data_valid; ++attempt) {
    std::memset(read_buffer.ptr_, 0, kBlobSize);
    auto get_task = cte_client->AsyncGetBlob(
        tag_id, blob_name, 0, kBlobSize, 0,
        read_buffer.shm_.template Cast<void>());
    get_task.Wait();
    if (get_task->GetReturnCode() != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    std::vector<char> read_data(kBlobSize);
    std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);
    data_valid = g_fixture->VerifyTestData(read_data, 'C');
  }
  REQUIRE(data_valid);

  CLIO_IPC->FreeBuffer(shm_buffer);
  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: cold blob demoted by the frecency organizer");
}

/**
 * Test: cleanup blobs and tag
 */
TEST_CASE("DataOrganizer - Cleanup", "[organizer][cleanup]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("organizer_test_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  auto del_hot = cte_client->AsyncDelBlob(tag_id, "hot_blob");
  del_hot.Wait();
  auto del_cold = cte_client->AsyncDelBlob(tag_id, "cold_blob");
  del_cold.Wait();
  auto del_tag = cte_client->AsyncDelTag("organizer_test_tag");
  del_tag.Wait();

  INFO("Cleanup complete");
}

int main(int argc, char **argv) {
  g_fixture = new DataOrganizerTestFixture();

  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  delete g_fixture;
  g_fixture = nullptr;

  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
