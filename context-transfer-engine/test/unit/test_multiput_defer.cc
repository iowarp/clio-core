/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * DEFERRED-PUT PIPELINE + MULTIPUT BATCH TEST (issue #862)
 *
 * Exercises the deferred-put pipeline end to end:
 *  - AsyncPutBlobDefer accumulates puts and ships 64-put MultiPutBlobTask
 *    batches (Method::kMultiPutBlob); the put count deliberately crosses
 *    batch boundaries and leaves a partial batch for the drain to flush.
 *  - Read-your-writes: AsyncGetBlobDefer must return the just-put bytes for
 *    keys whose puts are still pending (served from the accumulating batch
 *    or in-flight task, never stale/missing).
 *  - AwaitPutsUntilSpace(0) drains everything; afterwards every value must
 *    be durably readable through the normal GetBlob path and
 *    DeferErrorCount() must be zero.
 *  - AsyncMultiPutVectored is also driven directly with an explicit
 *    two-entry batch to pin the task's own contract (num_ok_, rc).
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

static constexpr int kNumPuts = 150;      // 2 full batches + a 22-put tail
static constexpr clio::run::u64 kValSize = 1024;

class MultiPutDeferFixture {
 public:
  std::string config_path_;

  MultiPutDeferFixture() {
    config_path_ = chi_test_data_dir() + "/multiput_defer_config.yaml";
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

  ~MultiPutDeferFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# MultiPut/defer test configuration - single 64MB DRAM tier
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
      - path: "ram::multiput_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

static std::string ValueFor(int i) {
  std::string v(kValSize, static_cast<char>('a' + (i % 26)));
  std::string prefix = "val_" + std::to_string(i);
  prefix.resize(16, '_');  // fixed-width so v keeps EXACTLY kValSize bytes
  v.replace(0, prefix.size(), prefix);
  return v;
}

TEST_CASE("MultiPutDefer - batched deferred puts with read-your-writes",
          "[cte][multiput][defer][862]") {
  MultiPutDeferFixture fixture;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("multiput_defer_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // 1. Defer kNumPuts values: crosses two 64-put batch boundaries and leaves
  //    a partial tail batch un-flushed.
  for (int i = 0; i < kNumPuts; ++i) {
    std::string val = ValueFor(i);
    int rc = client->AsyncPutBlobDefer(tag_id, "blob_" + std::to_string(i), 0,
                                       val.size(), val.data());
    REQUIRE(rc == 0);
  }

  // 2. Read-your-writes while puts are pending (some shipped, tail still
  //    accumulating): every value must come back exact.
  for (int i = 0; i < kNumPuts; i += 7) {
    std::string expect = ValueFor(i);
    std::vector<char> got(kValSize, 0);
    auto fut = client->AsyncGetBlobDefer(tag_id, "blob_" + std::to_string(i),
                                         0, kValSize, got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), expect.data(), kValSize) == 0);
  }

  // 3. Drain; no put may have failed.
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);

  // 4. Post-drain, every value is durably readable via the NORMAL path.
  for (int i = 0; i < kNumPuts; ++i) {
    std::string expect = ValueFor(i);
    std::vector<char> got(kValSize, 0);
    auto fut = client->AsyncGetBlob(tag_id, "blob_" + std::to_string(i), 0,
                                    kValSize, /*flags=*/0, got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), expect.data(), kValSize) == 0);
  }

  // 5. Overwrite a pending key twice, then read: newest-submitted must win
  //    (extent-set newest-wins compose).
  std::string v_old(kValSize, 'X');
  std::string v_new(kValSize, 'Y');
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "rewrite_me", 0, v_old.size(),
                                    v_old.data()) == 0);
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "rewrite_me", 0, v_new.size(),
                                    v_new.data()) == 0);
  {
    std::vector<char> got(kValSize, 0);
    auto fut =
        client->AsyncGetBlobDefer(tag_id, "rewrite_me", 0, kValSize, got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), v_new.data(), kValSize) == 0);
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
}

TEST_CASE("MultiPutDefer - large values via the recycled staging pool",
          "[cte][multiput][defer][892]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("multiput_defer_large_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Values >= the 128 KiB batch threshold take the LARGE defer path: staged
  // through the recycled pool (issue #892 — pool hits skip the allocator and
  // first-touch faults) and shipped as individual puts. Three rounds over
  // the same size class so later rounds RE-USE pooled buffers, with the
  // non-blocking completed-put reap feeding the pool mid-burst.
  constexpr clio::run::u64 kBig = 256 * 1024;
  constexpr int kRounds = 3;
  constexpr int kPerRound = 6;
  std::string big(kBig, 'L');
  for (int r = 0; r < kRounds; ++r) {
    for (int i = 0; i < kPerRound; ++i) {
      big[0] = static_cast<char>('a' + r);
      big[kBig - 1] = static_cast<char>('a' + i);
      int rc = client->AsyncPutBlobDefer(
          tag_id, "big_" + std::to_string(r) + "_" + std::to_string(i), 0,
          kBig, big.data());
      REQUIRE(rc == 0);
    }
    // Read-your-writes on a possibly-pending large put.
    {
      std::vector<char> got(kBig, 0);
      auto fut = client->AsyncGetBlobDefer(
          tag_id, "big_" + std::to_string(r) + "_0", 0, kBig, got.data());
      fut.Wait();
      REQUIRE(got[0] == static_cast<char>('a' + r));
    }
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);

  // Every value durably readable, byte-exact at both ends.
  for (int r = 0; r < kRounds; ++r) {
    for (int i = 0; i < kPerRound; ++i) {
      std::vector<char> got(kBig, 0);
      auto fut = client->AsyncGetBlob(
          tag_id, "big_" + std::to_string(r) + "_" + std::to_string(i), 0,
          kBig, /*flags=*/0, got.data());
      fut.Wait();
      REQUIRE(fut->GetReturnCode() == 0);
      REQUIRE(got[0] == static_cast<char>('a' + r));
      REQUIRE(got[kBig - 1] == static_cast<char>('a' + i));
    }
  }
}

TEST_CASE("MultiPutDefer - AsyncMultiPutVectored direct contract",
          "[cte][multiput][862]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);
  auto *ipc_manager = CLIO_CPU_IPC;

  clio::cte::core::Tag tag("multiput_direct_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Two payloads in ONE staged buffer + explicit descriptors.
  const std::string a(512, 'A'), b(768, 'B');
  ctp::ipc::FullPtr<char> staging =
      ipc_manager->AllocateBuffer(a.size() + b.size());
  REQUIRE(!staging.IsNull());
  std::memcpy(staging.ptr_, a.data(), a.size());
  std::memcpy(staging.ptr_ + a.size(), b.data(), b.size());

  std::vector<clio::cte::core::MultiPutDesc> descs(2);
  descs[0].tag_id_ = tag_id;
  descs[0].blob_name_ = "direct_a";
  descs[0].size_ = a.size();
  descs[0].payload_off_ = 0;
  descs[1].tag_id_ = tag_id;
  descs[1].blob_name_ = "direct_b";
  descs[1].size_ = b.size();
  descs[1].payload_off_ = a.size();

  auto fut = client->AsyncMultiPutVectored(
      ctp::ipc::ShmPtr<>(staging.shm_), a.size() + b.size(), descs);
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
  REQUIRE(fut->num_ok_ == 2);
  REQUIRE(fut->first_rc_ == 0);

  for (const auto &pair :
       std::vector<std::pair<std::string, const std::string *>>{
           {"direct_a", &a}, {"direct_b", &b}}) {
    std::vector<char> got(pair.second->size(), 0);
    auto g = client->AsyncGetBlob(tag_id, pair.first, 0, got.size(),
                                  /*flags=*/0, got.data());
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), pair.second->data(), got.size()) == 0);
  }
}

SIMPLE_TEST_MAIN()
