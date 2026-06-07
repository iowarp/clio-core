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
 * BLOB BLOCK-REUSE REGRESSION TEST
 *
 * Verifies the contract: PutBlob(1 MiB) followed by DelBlob frees the
 * block back to the SAME bdev allocator partition (size class) it was
 * drawn from, so the next PutBlob of the same size reuses it instead of
 * bump-allocating fresh space.
 *
 * Regression guard for the bug where CTE's FreeAllBlobBlocks passed
 * block_type_ = 0 to bdev FreeBlocks, filing every freed block in the
 * 4 KiB free list. AllocateBlock for 1 MiB looked in the 1 MiB list,
 * never found them, and fell through to the monotonic heap — so RAM
 * usage grew with op count regardless of the live key set and the tier
 * cap was hit after ~capacity bytes of *cumulative* (not live) writes.
 *
 * Method: a deliberately SMALL 256 MiB RAM tier; Put a fresh 1 MiB blob
 * then DelBlob it, kCycles = 1024 times. That is 1 GiB of *cumulative*
 * 1 MiB allocations — 4x the tier — but only ever ~1 MiB live at once.
 * With correct free-to-partition it runs forever; with the bug it
 * fails (out of space) after ~256 cycles. Every Put/Del must return 0.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = chi::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

static constexpr chi::u64 kBlobSize = 1 * 1024 * 1024;  // 1 MiB
// 1024 cycles * 1 MiB = 1 GiB cumulative, 4x the 256 MiB tier below.
// Overridable via BLOCK_REUSE_CYCLES: the CLIO_FORCE_NET stress variant routes
// every Put/Del through the loopback network path (≈300 ms/cycle), so 1024
// cycles would blow past the ctest timeout. Fewer cycles still exercise the
// free-to-partition net round-trip; the default keeps the tier-exhaustion
// regression at 4x the tier.
static int CyclesFromEnv() {
  if (const char *e = std::getenv("BLOCK_REUSE_CYCLES")) {
    int n = std::atoi(e);
    if (n > 0) return n;
  }
  return 1024;
}
static const int kCycles = CyclesFromEnv();

class BlockReuseFixture {
 public:
  std::string config_path_;
  bool initialized_ = false;

  BlockReuseFixture() {
    config_path_ = chi_test_data_dir() + "/block_reuse_config.yaml";
    Cleanup();
    CreateConfigFile();

    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    initialized_ = true;
  }

  ~BlockReuseFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }

  // Single small RAM tier (256 MiB). Small on purpose: if freed blocks
  // are not returned to the 1 MiB partition, cumulative 1 MiB allocations
  // exhaust it within ~256 Put/Del cycles.
  void CreateConfigFile() {
    std::ofstream f(config_path_);
    REQUIRE(f.is_open());
    f << R"(
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
      - path: "ram::block_reuse_tier"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    f.close();
  }
};

static BlockReuseFixture *g_fixture = nullptr;

/**
 * Put a fresh 1 MiB blob then DelBlob it, 1024x. Cumulative = 1 GiB,
 * 4x the 256 MiB tier; live set is always ~1 MiB. Passes only if every
 * DelBlob returns the block to the 1 MiB partition so the next PutBlob
 * reuses it (no monotonic heap growth → no out-of-space).
 */
TEST_CASE("BlockReuse - Put(1MB)+Del(1MB) frees back to its partition",
          "[blockreuse][regression]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto *cte = CLIO_CTE_CLIENT;
  REQUIRE(cte != nullptr);

  clio::cte::core::Tag tag("block_reuse_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  auto shm = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm.IsNull());
  std::memset(shm.ptr_, 0xAB, kBlobSize);
  ctp::ipc::ShmPtr<> shm_ptr = shm.shm_.template Cast<void>();

  int put_fail = 0;
  int del_fail = 0;
  int failed_cycle = -1;

  for (int i = 0; i < kCycles; ++i) {
    std::string blob_name = "reuse_blob_" + std::to_string(i);

    auto put = tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, 1.0f);
    put.Wait();
    if (put->GetReturnCode() != 0) {
      ++put_fail;
      if (failed_cycle < 0) failed_cycle = i;
      INFO("PutBlob failed at cycle " << i << " rc="
                                      << put->GetReturnCode()
                                      << " (cumulative MiB="
                                      << (i + 1) << ")");
      break;  // out of space: the regression
    }

    auto del = cte->AsyncDelBlob(tag_id, blob_name);
    del.Wait();
    if (del->GetReturnCode() != 0) {
      ++del_fail;
      if (failed_cycle < 0) failed_cycle = i;
      INFO("DelBlob failed at cycle " << i << " rc="
                                      << del->GetReturnCode());
      break;
    }
  }

  CLIO_IPC->FreeBuffer(shm);

  INFO("Completed Put/Del cycles before any failure: "
       << (failed_cycle < 0 ? kCycles : failed_cycle)
       << " / " << kCycles
       << "  (cumulative 1 MiB allocs would be "
       << (failed_cycle < 0 ? kCycles : failed_cycle)
       << " MiB vs 256 MiB tier)");

  // The whole point: 1 GiB cumulative through a 256 MiB tier only
  // succeeds if every freed 1 MiB block returns to the 1 MiB partition
  // and is reused. A regression manifests as an out-of-space PutBlob
  // around cycle ~256.
  REQUIRE(put_fail == 0);
  REQUIRE(del_fail == 0);

  auto del_tag = cte->AsyncDelTag("block_reuse_tag");
  del_tag.Wait();
}

int main(int argc, char **argv) {
  g_fixture = new BlockReuseFixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
