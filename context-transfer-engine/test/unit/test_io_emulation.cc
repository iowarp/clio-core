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
 * I/O EMULATION TEST (issue #747)
 *
 * Exercises Context::emulate_ on PutBlob/GetBlob:
 * - Emulated PutBlob allocates real placement (metadata/capacity effects)
 *   but skips the data write and returns a modeled duration in ns.
 * - Emulated GetBlob leaves the caller's buffer untouched and returns a
 *   modeled duration in ns; a subsequent real GetBlob still works.
 * - Bdev perf-stats persistence: file round trip via the static helpers,
 *   and stats files appearing under CLIO_BDEV_STATS_DIR from the running
 *   bdevs.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_runtime.h>
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

static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;  // 1MB

/**
 * Test fixture: two-tier CTE; bdev perf stats directed into the test dir.
 */
class IoEmulationTestFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  std::string bdev_stats_dir_;
  bool initialized_ = false;

  IoEmulationTestFixture() {
    INFO("=== Initializing IoEmulation Test ===");

    config_path_ = chi_test_data_dir() + "/io_emulation_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/io_emulation_storage.bin";
    bdev_stats_dir_ = chi_test_data_dir() + "/io_emulation_bdev_perf";

    Cleanup();
    CreateConfigFile();

    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    // Redirect bdev perf-stats persistence into the test directory so the
    // persistence assertions below can inspect it (and so this test never
    // pollutes the user's ~/.clio/bdev_perf).
    ctp::SystemInfo::Setenv("CLIO_BDEV_STATS_DIR", bdev_stats_dir_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    initialized_ = true;
    INFO("=== IoEmulation Test Environment Ready ===");
  }

  ~IoEmulationTestFixture() {
    INFO("=== Cleaning up IoEmulation Test ===");
    Cleanup();
  }

  void Cleanup() {
    std::error_code ec;
    if (fs::exists(config_path_)) fs::remove(config_path_, ec);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_, ec);
    fs::remove_all(bdev_stats_dir_, ec);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# IoEmulation Test Configuration
# - 16MB DRAM (fast tier, score 1.0)
# - 64MB File (slow tier, score 0.2)

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

    performance:
      stat_targets_period_ms: 1000

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 1000

    storage:
      # Fast tier: 16MB DRAM (score 1.0)
      - path: "ram::io_emulation_dram"
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
)";

    config_file.close();
    INFO("Created config file: " << config_path_);
  }
};

// Global fixture instance
static IoEmulationTestFixture *g_fixture = nullptr;

/**
 * Test: emulated PutBlob returns a modeled duration; placement is real
 */
TEST_CASE("IoEmulation - Emulated PutBlob", "[emulation][put][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("emulation_test_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "emulated_put_blob";

  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  std::memset(shm_buffer.ptr_, 'E', kBlobSize);

  auto put_task = cte_client->AsyncPutBlob(
      tag_id, blob_name, 0, kBlobSize,
      shm_buffer.shm_.template Cast<void>(), 1.0f,
      clio::cte::core::Context::Emulate(), 0);
  put_task.Wait();

  REQUIRE(put_task->GetReturnCode() == 0);
  clio::run::u64 put_ns = put_task->context_.emulated_time_ns_;
  INFO("emulated PutBlob time: " << put_ns << " ns");
  REQUIRE(put_ns > 0);

  // Metadata effects are real: the blob exists at the requested size...
  auto size_task = cte_client->AsyncGetBlobSize(tag_id, blob_name);
  size_task.Wait();
  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ == kBlobSize);

  // ...and a real GetBlob over the emulated blob succeeds (blocks are
  // allocated) even though the content was never written.
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());
  auto get_task = cte_client->AsyncGetBlob(
      tag_id, blob_name, 0, kBlobSize, 0,
      read_buffer.shm_.template Cast<void>());
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  CLIO_IPC->FreeBuffer(shm_buffer);
  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: emulated PutBlob");
}

/**
 * Test: emulated GetBlob models the time and leaves the buffer untouched
 */
TEST_CASE("IoEmulation - Emulated GetBlob", "[emulation][get][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("emulation_test_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "real_blob_for_emulated_get";

  // REAL put with a known pattern
  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  std::memset(shm_buffer.ptr_, 'R', kBlobSize);
  auto put_task = cte_client->AsyncPutBlob(
      tag_id, blob_name, 0, kBlobSize,
      shm_buffer.shm_.template Cast<void>(), 1.0f,
      clio::cte::core::Context(), 0);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  // A real put must not report an emulated time
  REQUIRE(put_task->context_.emulated_time_ns_ == 0);

  // EMULATED get: sentinel-filled buffer must come back untouched
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());
  std::memset(read_buffer.ptr_, 'S', kBlobSize);

  auto eget_task = cte_client->AsyncGetBlob(
      tag_id, blob_name, 0, kBlobSize, 0,
      read_buffer.shm_.template Cast<void>(),
      clio::run::PoolQuery::Dynamic(), clio::cte::core::Context::Emulate());
  eget_task.Wait();
  REQUIRE(eget_task->GetReturnCode() == 0);
  clio::run::u64 get_ns = eget_task->context_.emulated_time_ns_;
  INFO("emulated GetBlob time: " << get_ns << " ns");
  REQUIRE(get_ns > 0);

  bool untouched = true;
  const char *rb = reinterpret_cast<const char *>(read_buffer.ptr_);
  for (size_t i = 0; i < kBlobSize; ++i) {
    if (rb[i] != 'S') {
      untouched = false;
      break;
    }
  }
  REQUIRE(untouched);

  // A REAL get afterwards still returns the original pattern
  auto get_task = cte_client->AsyncGetBlob(
      tag_id, blob_name, 0, kBlobSize, 0,
      read_buffer.shm_.template Cast<void>());
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);
  bool pattern_ok = true;
  for (size_t i = 0; i < kBlobSize; ++i) {
    if (rb[i] != 'R') {
      pattern_ok = false;
      break;
    }
  }
  REQUIRE(pattern_ok);

  CLIO_IPC->FreeBuffer(shm_buffer);
  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: emulated GetBlob");
}

/**
 * Test: bdev perf-stats file save/load round trip (static helpers)
 */
TEST_CASE("IoEmulation - Perf Stats File Round Trip", "[emulation][perf]") {
  using clio::run::bdev::PerfMetrics;
  using BdevRuntime = clio::run::bdev::Runtime;

  std::string path =
      g_fixture->bdev_stats_dir_ + "/round_trip_test.perf";

  PerfMetrics out;
  out.read_bandwidth_mbps_ = 1234.5;
  out.write_bandwidth_mbps_ = 987.6;
  out.read_latency_us_ = 42.0;
  out.write_latency_us_ = 84.0;
  out.iops_ = 5000.0;

  REQUIRE(BdevRuntime::SavePerfStatsFile(path, out, 3.5f, 7.25f));

  PerfMetrics in;
  float wall_read = 0.0f;
  float wall_write = 0.0f;
  REQUIRE(BdevRuntime::LoadPerfStatsFile(path, in, wall_read, wall_write));
  REQUIRE(std::abs(in.read_bandwidth_mbps_ - 1234.5) < 1e-6);
  REQUIRE(std::abs(in.write_bandwidth_mbps_ - 987.6) < 1e-6);
  REQUIRE(std::abs(in.read_latency_us_ - 42.0) < 1e-6);
  REQUIRE(std::abs(in.write_latency_us_ - 84.0) < 1e-6);
  REQUIRE(std::abs(in.iops_ - 5000.0) < 1e-6);
  REQUIRE(std::abs(wall_read - 3.5f) < 1e-6f);
  REQUIRE(std::abs(wall_write - 7.25f) < 1e-6f);

  // Missing file / bad header are rejected
  PerfMetrics dummy;
  REQUIRE(!BdevRuntime::LoadPerfStatsFile(
      g_fixture->bdev_stats_dir_ + "/nope.perf", dummy, wall_read,
      wall_write));

  // Path derivation sanitizes non-path pool names under the stats dir
  std::string p = BdevRuntime::MakePerfStatsPath("ram::some_device");
  REQUIRE(p.rfind(g_fixture->bdev_stats_dir_, 0) == 0);
  REQUIRE(p.find("::") == std::string::npos);

  INFO("SUCCESS: perf stats file round trip");
}

/**
 * Test: the running bdevs persist their stats under CLIO_BDEV_STATS_DIR
 */
TEST_CASE("IoEmulation - Perf Stats Persisted By Runtime",
          "[emulation][perf][persist]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  // StatTargets (period 1000ms in this config) drives bdev GetStats, whose
  // first invocation persists a stats file per bdev (throttle starts
  // expired). Wait a few periods, then look for at least one file.
  bool found = false;
  for (int attempt = 0; attempt < 20 && !found; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::error_code ec;
    if (fs::exists(g_fixture->bdev_stats_dir_, ec)) {
      for (const auto &entry :
           fs::directory_iterator(g_fixture->bdev_stats_dir_, ec)) {
        if (entry.path().extension() == ".perf") {
          found = true;
          INFO("found persisted stats: " << entry.path().string());
          break;
        }
      }
    }
  }
  REQUIRE(found);
  INFO("SUCCESS: bdev perf stats persisted");
}

/**
 * Test: cleanup blobs and tag
 */
TEST_CASE("IoEmulation - Cleanup", "[emulation][cleanup]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("emulation_test_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  auto del1 = cte_client->AsyncDelBlob(tag_id, "emulated_put_blob");
  del1.Wait();
  auto del2 = cte_client->AsyncDelBlob(tag_id, "real_blob_for_emulated_get");
  del2.Wait();
  auto del_tag = cte_client->AsyncDelTag("emulation_test_tag");
  del_tag.Wait();

  INFO("Cleanup complete");
}

int main(int argc, char **argv) {
  g_fixture = new IoEmulationTestFixture();

  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  delete g_fixture;
  g_fixture = nullptr;

  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
