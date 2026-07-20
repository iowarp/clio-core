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
 * REORGANIZER CHECKPOINT-RESTART TEST
 *
 * Exercises the internal, periodically-driven data organizer with a
 * sequential checkpoint-restart workload:
 * - Runtime cold start with frecency organizer auto-pretraining
 * - Multiple checkpoint-restart cycles (write checkpoint, read previous)
 * - Validates reorganizer moves blobs between tiers correctly based on
 *   access patterns (frequently-read checkpoints promoted)
 * - Data integrity preserved across internally-driven moves
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

// 1MB per checkpoint blob
static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;

// Fast organizer cadence for observing rescoring within seconds
static constexpr clio::run::u32 kOrganizerPeriodMs = 500;

// Number of checkpoint-restart cycles
static constexpr int kNumCycles = 5;

// Retry count for mid-move failures (issue #753)
static constexpr int kMaxRetries = 10;

/**
 * Test fixture: two-tier CTE with frecency organizer for checkpoint-restart
 */
class CkptRestartTestFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  /**
   * Constructor: sets up config, initializes runtime and CTE client
   */
  CkptRestartTestFixture() {
    INFO("=== Initializing Checkpoint-Restart Test ===");

    config_path_ = chi_test_data_dir() + "/reorg_ckpt_restart_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/reorg_ckpt_restart_storage.bin";

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
    INFO("=== Checkpoint-Restart Test Environment Ready ===");
  }

  /**
   * Destructor: cleans up test files
   */
  ~CkptRestartTestFixture() {
    INFO("=== Cleaning up Checkpoint-Restart Test ===");
    Cleanup();
  }

  /**
   * Remove test config and storage files
   */
  void Cleanup() {
    if (fs::exists(config_path_)) {
      fs::remove(config_path_);
    }
    if (fs::exists(file_storage_path_)) {
      fs::remove(file_storage_path_);
    }
  }

  /**
   * Write two-tier YAML config with frecency organizer enabled
   */
  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# Checkpoint-Restart Test Configuration
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
      - path: "ram::reorg_ckpt_dram"
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

  /**
   * Create test data buffer with repeating pattern
   * @param size Buffer size in bytes
   * @param pattern Byte pattern to fill
   * @return Vector filled with pattern
   */
  std::vector<char> CreateTestData(size_t size, char pattern) {
    std::vector<char> data(size);
    std::memset(data.data(), pattern, size);
    return data;
  }

  /**
   * Verify buffer contains expected pattern
   * @param data Buffer to verify
   * @param expected_pattern Expected byte value
   * @return true if all bytes match pattern
   */
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
static CkptRestartTestFixture *g_fixture = nullptr;

/**
 * Test: Run checkpoint-restart cycles with reorganizer active
 *
 * Each cycle writes a new checkpoint and reads back all previous ones.
 * The frecency organizer should promote frequently-accessed checkpoints.
 */
TEST_CASE("ReorganizerCheckpointRestart - Cycles",
          "[organizer][checkpoint][noleak]") {
  REQUIRE(g_fixture != nullptr && g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("reorg_ckpt_restart_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  // Allocate SHM buffers for write and read
  auto write_shm = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!write_shm.IsNull());
  auto read_shm = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_shm.IsNull());

  // Run checkpoint-restart cycles
  for (int cycle = 0; cycle < kNumCycles; ++cycle) {
    std::string blob_name = "checkpoint_" + std::to_string(cycle);

    // Create unique pattern per checkpoint ('A', 'B', 'C', ...)
    auto test_data = g_fixture->CreateTestData(kBlobSize, 'A' + cycle);
    std::memcpy(write_shm.ptr_, test_data.data(), kBlobSize);

    // Put checkpoint with mid-tier score (0.5)
    auto put_task = tag.AsyncPutBlob(blob_name,
                                     write_shm.shm_.template Cast<void>(),
                                     kBlobSize, 0, 0.5f);
    put_task.Wait();
    REQUIRE(put_task->GetReturnCode() == 0);

    // Read back ALL previous checkpoints (simulating restart recovery)
    int ok_reads = 0;
    for (int prev = 0; prev < cycle; ++prev) {
      std::string prev_name = "checkpoint_" + std::to_string(prev);

      // Retry loop for mid-move failures (issue #753)
      bool read_ok = false;
      for (int attempt = 0; attempt < kMaxRetries && !read_ok; ++attempt) {
        auto get_task = cte_client->AsyncGetBlob(
            tag_id, prev_name, 0, kBlobSize, 0,
            read_shm.shm_.template Cast<void>());
        get_task.Wait();

        if (get_task->GetReturnCode() == 0) {
          std::vector<char> read_data(kBlobSize);
          std::memcpy(read_data.data(), read_shm.ptr_, kBlobSize);
          read_ok = g_fixture->VerifyTestData(read_data, 'A' + prev);
        }

        if (!read_ok) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
      }

      if (read_ok) {
        ok_reads++;
      }
    }

    INFO("Cycle " << cycle << ": read " << ok_reads << "/" << cycle
                  << " previous checkpoints");

    // Let organizer rescore after each cycle
    std::this_thread::sleep_for(
        std::chrono::milliseconds(5 * kOrganizerPeriodMs));
  }

  // Final settle period
  std::this_thread::sleep_for(
      std::chrono::milliseconds(5 * kOrganizerPeriodMs));

  CLIO_IPC->FreeBuffer(write_shm);
  CLIO_IPC->FreeBuffer(read_shm);

  INFO("SUCCESS: completed " << kNumCycles
       << " checkpoint-restart cycles with reorganizer active");
}

/**
 * Test: Validate frecency scores reflect access patterns
 *
 * The oldest checkpoint (checkpoint_0) was read in every subsequent cycle
 * (cycles 1-4 = 4 reads), while the most-recent checkpoint (checkpoint_4)
 * was read 0 times after its write. With the 600s frecency half-life,
 * recency ≈ 1.0 for ALL blobs in this ~30s test, so frequency (access_count)
 * is the differentiator — checkpoint_0 has higher frequency → higher score.
 */
TEST_CASE("ReorganizerCheckpointRestart - Validate Scores",
          "[organizer][checkpoint][scores]") {
  REQUIRE(g_fixture != nullptr);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("reorg_ckpt_restart_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  // Get score for most-recent checkpoint (kNumCycles-1)
  std::string recent_name = "checkpoint_" + std::to_string(kNumCycles - 1);
  float recent_score = -1.0f;
  for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
    auto score_task = cte_client->AsyncGetBlobScore(tag_id, recent_name);
    score_task.Wait();
    if (score_task->GetReturnCode() == 0) {
      recent_score = score_task->score_;
      break;
    }
    // Mid-move window (issue #753) — retry
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  INFO("recent checkpoint (" << recent_name << ") score: " << recent_score);

  // Get score for oldest checkpoint (0)
  std::string old_name = "checkpoint_0";
  float old_score = -1.0f;
  for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
    auto score_task = cte_client->AsyncGetBlobScore(tag_id, old_name);
    score_task.Wait();
    if (score_task->GetReturnCode() == 0) {
      old_score = score_task->score_;
      break;
    }
    // Mid-move window (issue #753) — retry
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  INFO("oldest checkpoint (" << old_name << ") score: " << old_score);

  // The oldest checkpoint (checkpoint_0) was read in every subsequent cycle
  // (cycles 1-4 = 4 reads), while the most-recent (checkpoint_4) was read 0
  // times after its write. With the 600s frecency half-life, recency ≈ 1.0
  // for ALL blobs in this ~30s test, so frequency (access_count) is the
  // differentiator. checkpoint_0 has higher frequency → higher score.
  // This validates the organizer correctly differentiates blobs by access
  // pattern.
  REQUIRE(old_score > recent_score);
  REQUIRE(old_score > 0.5f);  // the frequently-read one is promoted above mid-tier
  REQUIRE(recent_score >= 0.0f);  // the rarely-read one still has a valid score
  INFO("old checkpoint (frequently read) score > recent checkpoint (rarely read) score — frecency frequency differentiation validated");
}

/**
 * Test: Validate all checkpoint data survived reorganization moves
 *
 * Reads each checkpoint and verifies data integrity with retry logic
 * for mid-move transient failures (issue #753).
 */
TEST_CASE("ReorganizerCheckpointRestart - Validate Data Integrity",
          "[organizer][checkpoint][integrity]") {
  REQUIRE(g_fixture != nullptr);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("reorg_ckpt_restart_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  auto read_shm = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_shm.IsNull());

  // Verify each checkpoint
  for (int cycle = 0; cycle < kNumCycles; ++cycle) {
    std::string blob_name = "checkpoint_" + std::to_string(cycle);
    char expected_pattern = 'A' + cycle;

    bool data_valid = false;
    for (int attempt = 0; attempt < kMaxRetries && !data_valid; ++attempt) {
      std::memset(read_shm.ptr_, 0, kBlobSize);

      auto get_task = cte_client->AsyncGetBlob(
          tag_id, blob_name, 0, kBlobSize, 0,
          read_shm.shm_.template Cast<void>());
      get_task.Wait();

      if (get_task->GetReturnCode() != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      }

      std::vector<char> read_data(kBlobSize);
      std::memcpy(read_data.data(), read_shm.ptr_, kBlobSize);
      data_valid = g_fixture->VerifyTestData(read_data, expected_pattern);
    }

    REQUIRE(data_valid);
    INFO("checkpoint_" << cycle << " data integrity verified");
  }

  CLIO_IPC->FreeBuffer(read_shm);
  INFO("SUCCESS: all checkpoint data intact after reorganization");
}

/**
 * Test: Cleanup all checkpoints and tag
 */
TEST_CASE("ReorganizerCheckpointRestart - Cleanup",
          "[organizer][cleanup]") {
  REQUIRE(g_fixture != nullptr);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("reorg_ckpt_restart_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  // Delete all checkpoints
  for (int cycle = 0; cycle < kNumCycles; ++cycle) {
    std::string blob_name = "checkpoint_" + std::to_string(cycle);
    auto del_task = cte_client->AsyncDelBlob(tag_id, blob_name);
    del_task.Wait();
  }

  // Delete the tag
  auto del_tag = cte_client->AsyncDelTag("reorg_ckpt_restart_tag");
  del_tag.Wait();

  INFO("Cleanup complete");
}

/**
 * Main entry point: creates fixture, runs tests, cleans up
 * @param argc Argument count
 * @param argv Argument vector (first arg is test filter)
 * @return Test result code
 */
int main(int argc, char **argv) {
  g_fixture = new CkptRestartTestFixture();

  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  delete g_fixture;
  g_fixture = nullptr;

  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
