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
 *
 * Also reproduces issue #1116 (clio-fs wrong data: a 1 MiB page written as
 * eight 128 KiB AsyncPutBlobDefer partial puts can read back with one
 * sub-block aliased from another offset of the same blob). See the
 * "128 KiB partial puts" test cases below.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
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

// ==== Issue #1116 reproducer ================================================
//
// clio-fs reads back files through clio_cte_fuse with the correct size but
// wrong content. The suspected locus is cte_fuse_write splitting a write at
// 1 MiB page boundaries (kFsPageSize) and issuing one AsyncPutBlobDefer per
// 128 KiB slice (cp's write size, which is exactly
// DeferRegistry::kBatchChunk). AsyncPutBlobDefer's ordering guard only
// serializes OVERLAPPING [offset, offset+size) ranges for the same key —
// eight DISTINCT 128 KiB offsets of the SAME 1 MiB blob are never ordered
// against each other, so all eight partial puts for one page can be in
// flight concurrently against the server.
namespace repro1116 {

constexpr clio::run::u64 kPageSize = 1024ULL * 1024ULL;  // 1 MiB: kFsPageSize
// 128 KiB: cp's write size AND DeferRegistry::kBatchChunk, so every slice
// below takes the SAME "large" AsyncPutBlobDefer path the issue suspects.
constexpr clio::run::u64 kSubBlock = 128ULL * 1024ULL;
constexpr int kNumSubBlocks = static_cast<int>(kPageSize / kSubBlock);  // 8

/**
 * Fill a 1 MiB page buffer with per-128-KiB-sub-block content that is both
 * deterministic (reproducible from `generation` alone) and mutually
 * distinguishable, so a corrupted read can be matched back to whichever
 * sub-block's bytes actually landed there.
 *
 * Each sub-block's body comes from its own PRNG stream (seeded from
 * `generation` and the sub-block index) rather than a single repeated byte,
 * so a false "match" against the wrong sub-block cannot happen by
 * coincidence. The first bytes of each sub-block additionally carry a
 * human-readable "G<generation>_B<index>" tag purely for log messages.
 *
 * @param page output buffer, resized to kPageSize.
 * @param generation a per-call identifier (e.g. the test iteration index)
 *        that seeds every sub-block's content.
 */
void FillPage(std::vector<char> &page, clio::run::u64 generation) {
  page.resize(kPageSize);
  for (int k = 0; k < kNumSubBlocks; ++k) {
    std::mt19937_64 rng(generation * 1000003ULL +
                        static_cast<clio::run::u64>(k) * 7919ULL + 0x1116ULL);
    char *blk = page.data() + static_cast<clio::run::u64>(k) * kSubBlock;
    for (clio::run::u64 off = 0; off < kSubBlock; off += 8) {
      clio::run::u64 word = rng();
      clio::run::u64 n = std::min<clio::run::u64>(8, kSubBlock - off);
      std::memcpy(blk + off, &word, n);
    }
    char tag[24];
    std::snprintf(tag, sizeof(tag), "G%06llu_B%d__",
                  static_cast<unsigned long long>(generation), k);
    std::memcpy(blk, tag, std::min<size_t>(strlen(tag), kSubBlock));
  }
}

/**
 * Diagnose a mismatched 128 KiB sub-block by checking whether the bytes
 * actually read back exactly match one of `orig`'s OTHER sub-blocks — the
 * exact shape of corruption issue #1116 reports ("the bytes at page+128KiB
 * are the bytes from page+0").
 *
 * @param orig the page's expected (source) content.
 * @param got the page's actual content, as read back.
 * @param bad_k index of the sub-block that failed memcmp against `orig`.
 * @return a human-readable description naming the aliased source offset, or
 *         stating that no exact alias was found.
 */
std::string DescribeAliasing(const std::vector<char> &orig,
                             const std::vector<char> &got, int bad_k) {
  const char *got_blk = got.data() + static_cast<clio::run::u64>(bad_k) * kSubBlock;
  for (int j = 0; j < kNumSubBlocks; ++j) {
    const char *orig_blk = orig.data() + static_cast<clio::run::u64>(j) * kSubBlock;
    if (std::memcmp(got_blk, orig_blk, kSubBlock) == 0) {
      return "ALIASES this page's source sub-block " + std::to_string(j) +
             (j == bad_k ? " (identical to itself -- non-deterministic read)"
                         : "");
    }
  }
  return "no exact match against any of this page's 8 source sub-blocks "
         "(generic corruption, not a clean cross-offset alias)";
}

}  // namespace repro1116

TEST_CASE("MultiPutDefer - 128 KiB partial puts across a 1 MiB blob, "
          "shuffled vs sequential ordering (issue #1116)",
          "[cte][multiput][defer][1116]") {
  using namespace repro1116;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("multiput_1116_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Many "pages" (blobs), each filled by eight 128 KiB AsyncPutBlobDefer
  // calls submitted either sequential front-to-back (mimicking cp) or
  // shuffled (stresses the missing ordering guarantee directly), with
  // several pages' writes interleaved to mimic multiple files in flight.
  constexpr int kIterations = 200;    // keep total runtime well under ~2 min
  constexpr int kBlobsInFlight = 4;   // "multiple files" concurrently
  constexpr int kDrainEvery = 8;      // periodic explicit flush + durable check

  std::mt19937 order_rng(1116);
  int durable_mismatches = 0;
  int ryw_mismatches = 0;

  for (int iter = 0; iter < kIterations; ++iter) {
    int slot = iter % kBlobsInFlight;
    std::string blob_name = "page_" + std::to_string(slot);

    std::vector<char> orig;
    FillPage(orig, static_cast<clio::run::u64>(iter));

    std::vector<int> order(static_cast<size_t>(kNumSubBlocks));
    std::iota(order.begin(), order.end(), 0);
    bool sequential = (iter % 2 == 0);  // alternate cp-like vs shuffled
    if (!sequential) {
      std::shuffle(order.begin(), order.end(), order_rng);
    }

    // Eight partial puts, back-to-back, NOT awaited individually -- exactly
    // the pattern cte_fuse_write is suspected to drive.
    for (int k : order) {
      clio::run::u64 off = static_cast<clio::run::u64>(k) * kSubBlock;
      int rc = client->AsyncPutBlobDefer(tag_id, blob_name, off, kSubBlock,
                                         orig.data() + off);
      REQUIRE(rc == 0);
    }

    // Read-your-writes BEFORE any explicit flush: AsyncGetBlobDefer promises
    // to await/compose only the puts it needs, never missing or stale bytes.
    {
      std::vector<char> got(kPageSize, 0);
      auto fut = client->AsyncGetBlobDefer(tag_id, blob_name, 0, kPageSize,
                                           got.data());
      fut.Wait();
      if (std::memcmp(got.data(), orig.data(), kPageSize) != 0) {
        ryw_mismatches++;
        for (int k2 = 0; k2 < kNumSubBlocks; ++k2) {
          const char *g = got.data() + static_cast<clio::run::u64>(k2) * kSubBlock;
          const char *o = orig.data() + static_cast<clio::run::u64>(k2) * kSubBlock;
          if (std::memcmp(g, o, kSubBlock) != 0) {
            std::cout << "[#1116 repro][read-your-writes] iter=" << iter
                      << " slot=" << slot << " order="
                      << (sequential ? "sequential" : "shuffled")
                      << " sub-block " << k2 << " wrong: "
                      << DescribeAliasing(orig, got, k2) << std::endl;
          }
        }
      }
    }

    // Periodically drain everything explicitly and verify durability via the
    // NORMAL (non-defer) read path -- the state the issue's `cp` + sha256sum
    // repro actually observes.
    if ((iter + 1) % kDrainEvery == 0 || iter + 1 == kIterations) {
      clio::cte::core::Client::AwaitPutsUntilSpace(0);
      REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);

      for (int s = 0; s < kBlobsInFlight; ++s) {
        int last_iter = -1;
        for (int cand = iter; cand >= 0; --cand) {
          if (cand % kBlobsInFlight == s) { last_iter = cand; break; }
        }
        if (last_iter < 0) continue;
        std::vector<char> expect;
        FillPage(expect, static_cast<clio::run::u64>(last_iter));
        std::vector<char> got(kPageSize, 0);
        auto fut = client->AsyncGetBlob(tag_id, "page_" + std::to_string(s),
                                        0, kPageSize, /*flags=*/0,
                                        got.data());
        fut.Wait();
        REQUIRE(fut->GetReturnCode() == 0);
        if (std::memcmp(got.data(), expect.data(), kPageSize) != 0) {
          durable_mismatches++;
          for (int k2 = 0; k2 < kNumSubBlocks; ++k2) {
            const char *g = got.data() + static_cast<clio::run::u64>(k2) * kSubBlock;
            const char *o = expect.data() + static_cast<clio::run::u64>(k2) * kSubBlock;
            if (std::memcmp(g, o, kSubBlock) != 0) {
              std::cout << "[#1116 repro][durable] iter=" << iter
                        << " slot=" << s << " (last write iter=" << last_iter
                        << ") sub-block " << k2 << " wrong: "
                        << DescribeAliasing(expect, got, k2) << std::endl;
            }
          }
        }
      }
    }
  }

  std::cout << "[#1116 repro] " << kIterations << " pages, " << kBlobsInFlight
            << " in flight: " << durable_mismatches
            << " durable mismatches, " << ryw_mismatches
            << " read-your-write mismatches" << std::endl;
  REQUIRE(durable_mismatches == 0);
  REQUIRE(ryw_mismatches == 0);
}

TEST_CASE("MultiPutDefer - ~1 GiB cumulative traffic through a small RAM "
          "tier, 128 KiB partial puts (issue #1116 stress)",
          "[cte][multiput][defer][1116]") {
  using namespace repro1116;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("multiput_1116_stress_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Same access pattern as the case above, but driven much harder: this
  // test case runs AFTER the rest of the file's traffic (same 64 MiB DRAM
  // tier, already exercised by every earlier TEST_CASE in this binary), and
  // itself pushes roughly 1 GiB cumulative payload through a small, reused
  // set of blob slots -- matching the issue's observation that the failure
  // correlates with prior tier traffic and the recycled staging pool
  // (issue #892), not with tier exhaustion.
  constexpr int kIterations = 1000;  // 1000 * 8 * 128 KiB ~= 1000 MiB
  constexpr int kBlobsInFlight = 4;
  constexpr int kDrainEvery = 32;

  std::mt19937 order_rng(892111601);
  int durable_mismatches = 0;

  for (int iter = 0; iter < kIterations; ++iter) {
    int slot = iter % kBlobsInFlight;
    std::string blob_name = "stress_page_" + std::to_string(slot);

    std::vector<char> orig;
    FillPage(orig, static_cast<clio::run::u64>(iter) + 0x7000000ULL);

    std::vector<int> order(static_cast<size_t>(kNumSubBlocks));
    std::iota(order.begin(), order.end(), 0);
    if (iter % 2 != 0) {
      std::shuffle(order.begin(), order.end(), order_rng);
    }
    for (int k : order) {
      clio::run::u64 off = static_cast<clio::run::u64>(k) * kSubBlock;
      int rc = client->AsyncPutBlobDefer(tag_id, blob_name, off, kSubBlock,
                                         orig.data() + off);
      REQUIRE(rc == 0);
    }

    if ((iter + 1) % kDrainEvery == 0 || iter + 1 == kIterations) {
      clio::cte::core::Client::AwaitPutsUntilSpace(0);
      REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);

      for (int s = 0; s < kBlobsInFlight; ++s) {
        int last_iter = -1;
        for (int cand = iter; cand >= 0; --cand) {
          if (cand % kBlobsInFlight == s) { last_iter = cand; break; }
        }
        if (last_iter < 0) continue;
        std::vector<char> expect;
        FillPage(expect, static_cast<clio::run::u64>(last_iter) + 0x7000000ULL);
        std::vector<char> got(kPageSize, 0);
        auto fut = client->AsyncGetBlob(
            tag_id, "stress_page_" + std::to_string(s), 0, kPageSize,
            /*flags=*/0, got.data());
        fut.Wait();
        REQUIRE(fut->GetReturnCode() == 0);
        if (std::memcmp(got.data(), expect.data(), kPageSize) != 0) {
          durable_mismatches++;
          for (int k2 = 0; k2 < kNumSubBlocks; ++k2) {
            const char *g = got.data() + static_cast<clio::run::u64>(k2) * kSubBlock;
            const char *o = expect.data() + static_cast<clio::run::u64>(k2) * kSubBlock;
            if (std::memcmp(g, o, kSubBlock) != 0) {
              std::cout << "[#1116 stress][durable] iter=" << iter
                        << " slot=" << s << " (last write iter=" << last_iter
                        << ") sub-block " << k2 << " wrong: "
                        << DescribeAliasing(expect, got, k2) << std::endl;
            }
          }
        }
      }
    }
  }

  std::cout << "[#1116 stress] " << kIterations << " pages (~"
            << (kIterations * kPageSize / (1024 * 1024)) << " MiB traffic): "
            << durable_mismatches << " durable mismatches" << std::endl;
  REQUIRE(durable_mismatches == 0);
}

SIMPLE_TEST_MAIN()
