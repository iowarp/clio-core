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
 * REORGANIZE HINT TEST
 *
 * Exercises Client::AsyncReorganizeHint / ReorganizeHint (Method::kReorganizeHint):
 * the application-facing knob that tells the data organizer which phase of its
 * algorithm is running. The core treats the value as opaque and stores it per
 * container (Runtime::organizer_hint_); this test checks the task is accepted,
 * broadcast completes with return code 0, and repeated hints (including
 * negative and zero) are all accepted -- the value's meaning belongs to the
 * organizer, so there is nothing else for the core to assert.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

class ReorganizeHintFixture {
 public:
  std::string config_path_;

  ReorganizeHintFixture() {
    config_path_ = chi_test_data_dir() + "/reorganize_hint_config.yaml";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  ~ReorganizeHintFixture() { Cleanup(); }

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
};

TEST_CASE("ReorganizeHint is accepted and broadcast completes", "[cte][reorganize_hint]") {
  ReorganizeHintFixture fx;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  SECTION("a positive phase hint") {
    auto task = client->AsyncReorganizeHint(3);
    task.Wait();
    REQUIRE(task->GetReturnCode() == 0);
  }
  SECTION("the synchronous wrapper returns the task's code") {
    REQUIRE(client->ReorganizeHint(7) == 0);
  }
  SECTION("zero and negative hints are just values, not errors") {
    REQUIRE(client->ReorganizeHint(0) == 0);
    REQUIRE(client->ReorganizeHint(-1) == 0);
  }
  SECTION("repeated hints in sequence all complete") {
    for (clio::run::i32 phase = 1; phase <= 5; ++phase) {
      REQUIRE(client->ReorganizeHint(phase) == 0);
    }
  }
}

SIMPLE_TEST_MAIN()
