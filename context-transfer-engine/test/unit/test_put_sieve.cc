/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CLIENT-SIDE DATA SIEVING TEST (issue #1007)
 *
 * AsyncPutBlobDefer coalesces small partial-object writes into per-blob
 * 64 KiB pages before they become tasks. This exercises the sieve's whole
 * contract:
 *  - Sequential small writes coalesce and drain to byte-exact data.
 *  - Read-your-writes serves from an OPEN page without draining it.
 *  - In-place overwrite of unshipped bytes: newest write wins.
 *  - An intra-page hole sweeps the old extent and re-opens the page (v1
 *    keeps ONE contiguous extent per page).
 *  - A write spanning a page boundary splits across pages.
 *  - The background flusher ships idle partial pages with NO drain call.
 *  - Random overlapping small writes converge to the newest-wins image.
 *
 * The same binary registered with CLIO_CTE_PUT_SIEVE=0 is the kill-switch
 * parity run: every case must also pass through the plain batch pipeline.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
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

class PutSieveFixture {
 public:
  std::string config_path_;

  PutSieveFixture() {
    config_path_ = chi_test_data_dir() + "/put_sieve_config.yaml";
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

  ~PutSieveFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Put-sieve test configuration - single 64MB DRAM tier
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
      - path: "ram::put_sieve_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

/** Deterministic byte for (blob offset, generation). */
static char PatByte(clio::run::u64 off, int gen) {
  return static_cast<char>('A' + ((off / 512 + static_cast<clio::run::u64>(gen) * 7) % 26));
}

static void FillPat(std::vector<char> &buf, clio::run::u64 off, int gen) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = PatByte(off + i, gen);
  }
}

TEST_CASE("PutSieve - sequential small writes coalesce and read back",
          "[cte][sieve][1007]") {
  PutSieveFixture fixture;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_seq_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // 256 x 512 B sequential writes = 128 KiB: two full 64 KiB pages' worth.
  // (The very first write is offset 0 to a fresh key, so it takes the batch
  // path by design; everything after it sieves.)
  constexpr clio::run::u64 kPiece = 512;
  constexpr int kPieces = 256;
  std::vector<char> piece(kPiece);
  for (int i = 0; i < kPieces; ++i) {
    clio::run::u64 off = static_cast<clio::run::u64>(i) * kPiece;
    FillPat(piece, off, 0);
    REQUIRE(client->AsyncPutBlobDefer(tag_id, "stream", off, kPiece,
                                      piece.data()) == 0);
    // Mid-stream read-your-writes from the OPEN page, once page 2 has a few
    // pieces: must be served byte-exact without disturbing the stream.
    if (i == 140) {
      std::vector<char> got(kPiece, 0);
      auto fut = client->AsyncGetBlobDefer(tag_id, "stream", 137 * kPiece,
                                           kPiece, got.data());
      fut.Wait();
      std::vector<char> expect(kPiece);
      FillPat(expect, 137 * kPiece, 0);
      REQUIRE(std::memcmp(got.data(), expect.data(), kPiece) == 0);
    }
  }

  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);

  // The whole 128 KiB must be durably byte-exact via the NORMAL read path.
  std::vector<char> all(kPieces * kPiece, 0);
  auto fut = client->AsyncGetBlob(tag_id, "stream", 0, all.size(),
                                  /*flags=*/0, all.data());
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
  std::vector<char> expect(all.size());
  FillPat(expect, 0, 0);
  REQUIRE(std::memcmp(all.data(), expect.data(), all.size()) == 0);
}

TEST_CASE("PutSieve - in-place overwrite of an open page, newest wins",
          "[cte][sieve][1007]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_rw_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  std::string v_old(512, 'X');
  std::string v_new(512, 'Y');
  // Non-zero offset -> sieved; the second write lands IN PLACE on the same
  // open page (one extent, one eventual put) and must win.
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "rw", 512, v_old.size(),
                                    v_old.data()) == 0);
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "rw", 512, v_new.size(),
                                    v_new.data()) == 0);
  {
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlobDefer(tag_id, "rw", 512, 512, got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), v_new.data(), 512) == 0);
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
  {
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlob(tag_id, "rw", 512, 512, /*flags=*/0,
                                    got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v_new.data(), 512) == 0);
  }
}

TEST_CASE("PutSieve - intra-page hole sweeps and re-opens the page",
          "[cte][sieve][1007]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_hole_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  std::string a(512, 'H');
  std::string b(512, 'G');
  // [512,1024) then [4096,4608): same 64 KiB page, non-contiguous -> the
  // first extent is swept out and the page re-opens for the second.
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "hole", 512, a.size(),
                                    a.data()) == 0);
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "hole", 4096, b.size(),
                                    b.data()) == 0);
  // Both writes must be readable pre-drain (the swept extent now serves from
  // the batch chunk; the open page serves the second).
  {
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlobDefer(tag_id, "hole", 512, 512, got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), a.data(), 512) == 0);
  }
  {
    std::vector<char> got(512, 0);
    auto fut =
        client->AsyncGetBlobDefer(tag_id, "hole", 4096, 512, got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), b.data(), 512) == 0);
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
  {
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlob(tag_id, "hole", 512, 512, /*flags=*/0,
                                    got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), a.data(), 512) == 0);
  }
  {
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlob(tag_id, "hole", 4096, 512, /*flags=*/0,
                                    got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), b.data(), 512) == 0);
  }
}

TEST_CASE("PutSieve - a write spanning a page boundary splits across pages",
          "[cte][sieve][1007]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_span_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // [65200, 66200): crosses the 64 KiB page boundary at 65536.
  constexpr clio::run::u64 kOff = 65200;
  constexpr clio::run::u64 kLen = 1000;
  std::vector<char> v(kLen);
  FillPat(v, kOff, 3);
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "span", kOff, kLen, v.data()) ==
          0);
  {
    std::vector<char> got(kLen, 0);
    auto fut =
        client->AsyncGetBlobDefer(tag_id, "span", kOff, kLen, got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), v.data(), kLen) == 0);
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
  {
    std::vector<char> got(kLen, 0);
    auto fut = client->AsyncGetBlob(tag_id, "span", kOff, kLen, /*flags=*/0,
                                    got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v.data(), kLen) == 0);
  }
}

TEST_CASE("PutSieve - background flusher ships idle partial pages unaided",
          "[cte][sieve][1007]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_idle_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  std::string v(512, 'F');
  REQUIRE(client->AsyncPutBlobDefer(tag_id, "idle", 512, v.size(),
                                    v.data()) == 0);
  bool landed = false;
  if (clio::cte::core::Client::SievePutEnabled()) {
    // NO drain call of any kind: the 500 us flusher must sweep the idle page
    // and ship it. Poll the NORMAL read path (which never consults pending
    // writes) until the bytes land. Generous bound for loaded CI machines.
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      std::vector<char> got(512, 0);
      auto fut = client->AsyncGetBlob(tag_id, "idle", 512, 512, /*flags=*/0,
                                      got.data());
      fut.Wait();
      if (fut->GetReturnCode() == 0 &&
          std::memcmp(got.data(), v.data(), 512) == 0) {
        landed = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  } else {
    // Kill-switch parity: with CLIO_CTE_PUT_SIEVE=0 there is no flusher,
    // but there also is no page — the write sits in the accumulating batch.
    // Flush explicitly so the case is meaningful in this mode too.
    clio::cte::core::Client::AwaitPutsUntilSpace(0);
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlob(tag_id, "idle", 512, 512, /*flags=*/0,
                                    got.data());
    fut.Wait();
    landed = fut->GetReturnCode() == 0 &&
             std::memcmp(got.data(), v.data(), 512) == 0;
  }
  REQUIRE(landed);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
}

TEST_CASE("PutSieve - per-blob budget sweeps the oldest page inline",
          "[cte][sieve][1007]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_budget_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // 21 writes, each in its OWN 64 KiB page of one blob: exceeds the 16-page
  // (1 MiB) per-blob budget, so creating pages 17..21 sweeps the blob's
  // oldest page inline. Every write must remain readable BOTH pre-drain
  // (swept extents serve from the batch chunk, open ones from their pages)
  // and post-drain.
  constexpr int kPages = 21;
  std::vector<char> blk(512);
  for (int i = 0; i < kPages; ++i) {
    clio::run::u64 off = 1024 + static_cast<clio::run::u64>(i) * 65536;
    FillPat(blk, off, 9);
    REQUIRE(client->AsyncPutBlobDefer(tag_id, "budget", off, blk.size(),
                                      blk.data()) == 0);
  }
  for (int i = 0; i < kPages; i += 4) {
    clio::run::u64 off = 1024 + static_cast<clio::run::u64>(i) * 65536;
    std::vector<char> got(512, 0);
    auto fut =
        client->AsyncGetBlobDefer(tag_id, "budget", off, 512, got.data());
    fut.Wait();
    std::vector<char> expect(512);
    FillPat(expect, off, 9);
    REQUIRE(std::memcmp(got.data(), expect.data(), 512) == 0);
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
  for (int i = 0; i < kPages; ++i) {
    clio::run::u64 off = 1024 + static_cast<clio::run::u64>(i) * 65536;
    std::vector<char> got(512, 0);
    auto fut = client->AsyncGetBlob(tag_id, "budget", off, 512, /*flags=*/0,
                                    got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    std::vector<char> expect(512);
    FillPat(expect, off, 9);
    REQUIRE(std::memcmp(got.data(), expect.data(), 512) == 0);
  }
}

TEST_CASE("PutSieve - random overlapping small writes converge",
          "[cte][sieve][1007]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("sieve_rand_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // 400 writes of 512 B at random 512-aligned offsets in a 256 KiB region:
  // overlaps land newest-wins, holes churn the sweep path, page count churns
  // the cap. Mirror every write into a local image; only written blocks are
  // compared (unwritten gaps inside the blob are unspecified).
  constexpr clio::run::u64 kRegion = 256 * 1024;
  constexpr clio::run::u64 kBlk = 512;
  constexpr int kWrites = 400;
  std::vector<char> image(kRegion, 0);
  std::vector<int> writes(kRegion / kBlk, 0);
  std::mt19937 rng(10071007);  // fixed seed: reproducible
  std::uniform_int_distribution<clio::run::u64> pick(0, kRegion / kBlk - 1);
  std::vector<char> blk(kBlk);
  for (int i = 0; i < kWrites; ++i) {
    clio::run::u64 b = pick(rng);
    clio::run::u64 off = b * kBlk;
    FillPat(blk, off, i);
    std::memcpy(image.data() + off, blk.data(), kBlk);
    writes[b]++;
    REQUIRE(client->AsyncPutBlobDefer(tag_id, "rand", off, kBlk,
                                      blk.data()) == 0);
  }
  // Pre-drain read-your-writes on once-written blocks (an overlapped block
  // whose earlier task already retired reads the durable value, which is
  // scheduler-ordered — same reason as the post-drain restriction below).
  for (clio::run::u64 b = 0; b < kRegion / kBlk; b += 5) {
    if (writes[b] != 1) continue;
    std::vector<char> got(kBlk, 0);
    auto fut = client->AsyncGetBlobDefer(tag_id, "rand", b * kBlk, kBlk,
                                         got.data());
    fut.Wait();
    REQUIRE(std::memcmp(got.data(), image.data() + b * kBlk, kBlk) == 0);
  }
  clio::cte::core::Client::AwaitPutsUntilSpace(0);
  REQUIRE(clio::cte::core::Client::DeferErrorCount() == 0);
  // Post-drain durability is compared for ONCE-written blocks only: two
  // deferred writes to the same bytes that end up in different tasks are
  // applied in scheduler order — the pipeline's documented (pre-existing)
  // contract, which the batch path shares.
  for (clio::run::u64 b = 0; b < kRegion / kBlk; ++b) {
    if (writes[b] != 1) continue;
    std::vector<char> got(kBlk, 0);
    auto fut = client->AsyncGetBlob(tag_id, "rand", b * kBlk, kBlk,
                                    /*flags=*/0, got.data());
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), image.data() + b * kBlk, kBlk) == 0);
  }
}

SIMPLE_TEST_MAIN()
