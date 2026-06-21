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
 * TIERED STORAGE STRESS TEST — "0g" DEFAULT (80% DRAM) RAM TIER
 *
 * Regression + stress coverage for the policy:
 *   a RAM bdev / CTE storage tier configured with capacity "0g" (or 0)
 *   defaults to 80% of total system DRAM (clio::run::bdev::
 *   DefaultRamCapacityBytes()) instead of being rejected (old CTE
 *   behavior) or treated as unbounded (old bdev behavior).
 *
 * Topology:
 *   - Fast tier: ram::dram_default, capacity_limit "0g"  -> 80% DRAM,
 *     score 1.0
 *   - Slow tier: small bounded file, capacity_limit 64MB, score 0.0
 *
 * What is exercised:
 *   1. CTE config parsing ACCEPTS "0g" for a ram tier (previously a
 *      hard error) and the runtime composes successfully.
 *   2. The "0g" RAM tier is real and usable: a multi-hundred-MB working
 *      set is Put through the DPE/tiering path and lands successfully
 *      (an unbounded or zero-sized tier would mis-place or OOM).
 *   3. The tiering/migration path is stressed: ReorganizeBlob moves the
 *      whole dataset down to the constrained 64MB file tier and back up
 *      to the "0g" DRAM tier.
 *   4. Data integrity is preserved across all migrations.
 *
 * The working set is intentionally small relative to 80% of DRAM so the
 * test is safe on RAM-constrained CI runners while still driving real
 * placement decisions and inter-tier migration.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_tasks.h>  // clio::run::bdev::DefaultRamCapacityBytes
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

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

// Working set: small vs. 80% DRAM, big enough to drive real placement +
// inter-tier migration.
static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;          // 1MB
static constexpr clio::run::u64 kTotalDataSize = 96 * 1024 * 1024;    // 96MB
static constexpr int kNumBlobs = kTotalDataSize / kBlobSize;    // 96 blobs
static constexpr clio::run::u64 kSlowFileCapacity = 64 * 1024 * 1024; // 64MB

static constexpr float kFastTierScore = 1.0f;  // ram::dram_default (0g)
static constexpr float kSlowTierScore = 0.0f;  // bounded file tier

class DramDefaultTieringFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  DramDefaultTieringFixture() {
    INFO("=== Initializing 0g-default (80% DRAM) Tiering Stress Test ===");

    config_path_ = chi_test_data_dir() + "/dram_default_tiering_config.yaml";
    file_storage_path_ =
        chi_test_data_dir() + "/dram_default_tiering_storage.bin";

    Cleanup();
    CreateConfigFile();

    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    initialized_ = true;
    INFO("=== Environment Ready ===");
  }

  ~DramDefaultTieringFixture() {
    INFO("=== Cleaning up 0g-default Tiering Stress Test ===");
    Cleanup();
  }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_);
  }

  /**
   * Fast tier deliberately uses capacity_limit "0g" — the policy under
   * test. Slow tier is a bounded 64MB file so reorganization down to it
   * exercises constrained-capacity handling.
   */
  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# 0g-default (80% DRAM) tiering stress configuration
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
      # Fast tier: RAM, capacity "0g" -> 80% of total system DRAM
      - path: "ram::dram_default"
        bdev_type: "ram"
        capacity_limit: "0g"
        score: 1.0

      # Slow tier: bounded 64MB file
      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 0.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
    INFO("Created config file: " << config_path_);
  }

  std::vector<char> CreateTestData(size_t size, int blob_index) {
    std::vector<char> data(size);
    char pattern = static_cast<char>('A' + (blob_index % 26));
    std::memset(data.data(), pattern, size);
    return data;
  }

  bool VerifyTestData(const std::vector<char> &data, int blob_index) {
    char expected = static_cast<char>('A' + (blob_index % 26));
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] != expected) return false;
    }
    return true;
  }
};

static DramDefaultTieringFixture *g_fixture = nullptr;

/**
 * Sanity-check the policy math itself: the shared helper that both the
 * bdev module and CTE use must report a sane, non-zero capacity that
 * comfortably exceeds this test's working set (otherwise the "0g" tier
 * could not absorb the data and the rest of the suite would be
 * meaningless).
 */
TEST_CASE("DramDefault - 80% policy is sane and >= working set",
          "[tiered][dram-default][policy]") {
  clio::run::u64 total_dram = ctp::SystemInfo::GetRamCapacity();
  clio::run::u64 defaulted = clio::run::bdev::DefaultRamCapacityBytes();

  INFO("Total system DRAM: " << total_dram << " bytes");
  INFO("0g default (80%):  " << defaulted << " bytes");

  REQUIRE(total_dram > 0);
  REQUIRE(defaulted > 0);
  REQUIRE(defaulted < total_dram);  // strictly a fraction, not unbounded

  // ~80% (allow rounding slack from the double multiply).
  double frac = static_cast<double>(defaulted) /
                static_cast<double>(total_dram);
  REQUIRE(frac > 0.78);
  REQUIRE(frac < 0.82);

  // Must dwarf the test working set so the fast tier is genuinely usable.
  REQUIRE(defaulted > kTotalDataSize);
}

/**
 * Put the full working set. With the fast tier resolved from "0g" to
 * 80% DRAM (>> 96MB) and a high score, placement must succeed for every
 * blob — proving the defaulted RAM tier is real and addressable.
 */
TEST_CASE("DramDefault - Put 96MB into 0g RAM tier",
          "[tiered][dram-default][put]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("dram_default_tag");

  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  ctp::ipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

  int success_count = 0;
  int failure_count = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    auto test_data = g_fixture->CreateTestData(kBlobSize, i);
    std::memcpy(shm_buffer.ptr_, test_data.data(), kBlobSize);

    auto put_task =
        tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, kFastTierScore);
    put_task.Wait();
    if (put_task->GetReturnCode() == 0) {
      success_count++;
    } else {
      failure_count++;
      INFO("Put failed for blob " << i << " rc="
                                  << put_task->GetReturnCode());
    }
  }
  CLIO_IPC->FreeBuffer(shm_buffer);

  INFO("Put results: " << success_count << " ok, " << failure_count
                       << " failed");
  REQUIRE(failure_count == 0);
  REQUIRE(success_count == kNumBlobs);
}

/**
 * Stress the tiering/migration path: push the whole dataset DOWN to the
 * constrained 64MB file tier, then pull it back UP to the "0g" DRAM
 * tier. Exercises constrained-capacity handling on the way down and the
 * defaulted-capacity tier as a migration destination on the way up.
 */
TEST_CASE("DramDefault - Reorganize down to file then up to 0g RAM",
          "[tiered][dram-default][reorganize]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("dram_default_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  // Down to slow (bounded 64MB file) tier.
  int down_ok = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    auto t = cte_client->AsyncReorganizeBlob(tag_id, blob_name,
                                             kSlowTierScore);
    t.Wait();
    if (t->GetReturnCode() == 0) down_ok++;
  }
  INFO("Reorganize down (-> file): " << down_ok << "/" << kNumBlobs);
  REQUIRE(down_ok == kNumBlobs);

  // Back up to the "0g" (80% DRAM) fast tier.
  int up_ok = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    auto t = cte_client->AsyncReorganizeBlob(tag_id, blob_name,
                                             kFastTierScore);
    t.Wait();
    if (t->GetReturnCode() == 0) up_ok++;
  }
  INFO("Reorganize up (-> 0g RAM): " << up_ok << "/" << kNumBlobs);
  REQUIRE(up_ok == kNumBlobs);
}

/**
 * Data must survive both migrations intact.
 */
TEST_CASE("DramDefault - Verify integrity after migration",
          "[tiered][dram-default][integrity]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  clio::cte::core::Tag tag("dram_default_tag");

  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  int verified = 0;
  int corrupted = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    tag.GetBlob(blob_name, read_buffer.shm_.template Cast<void>(), kBlobSize,
                0);
    std::vector<char> read_data(kBlobSize);
    std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);
    if (g_fixture->VerifyTestData(read_data, i)) {
      verified++;
    } else {
      corrupted++;
      INFO("Corruption in blob " << i);
    }
  }
  CLIO_IPC->FreeBuffer(read_buffer);

  INFO("Integrity: " << verified << " ok, " << corrupted << " corrupted");
  REQUIRE(corrupted == 0);
  REQUIRE(verified == kNumBlobs);
}

/**
 * Tear down all blobs + the tag.
 */
TEST_CASE("DramDefault - Cleanup", "[tiered][dram-default][cleanup]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("dram_default_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    auto t = cte_client->AsyncDelBlob(tag_id, blob_name);
    t.Wait();
  }
  auto del_tag = cte_client->AsyncDelTag("dram_default_tag");
  del_tag.Wait();
  INFO("Cleanup complete");
}

int main(int argc, char **argv) {
  g_fixture = new DramDefaultTieringFixture();

  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  delete g_fixture;
  g_fixture = nullptr;
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
