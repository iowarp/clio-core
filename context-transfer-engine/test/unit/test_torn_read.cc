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

// Torn-read regression (RELIABILITY.md defects 8 and 11).
//
// Readers of a blob that is being overwritten IN PLACE must see a whole
// version, never a mixture. Before the fixes:
//  - the RPC path: PutBlob took the write token but never drained pinned
//    readers, so a GetBlob mid-ReadData returned old and new bytes together;
//  - the zero-IPC path: TryReadBlobShm memcpy'd extents straight out of the
//    RAM tier and re-validated only placement_gen_, which an in-place put
//    does not change.
// Both were measured by clio_cte_vector_stress (pages with the new
// generation's header and the old generation's tail, rc=0).
//
// Every page carries {blob, generation} in its first two words and a hash of
// (blob, generation, index) in the rest, so a reader can tell a torn page from
// any whole one without knowing which generation it should get. Writers
// overwrite K blobs continuously while readers get them; a page whose words
// disagree with its own header is a torn read and fails the test.
//
// Case 1 uses generational gets (kGenerational): the runtime path, read
// pinning against the writer's drain. Case 2 uses plain gets from this
// process: the client's shared-memory fast path with the content seqlock.
// The two cases together cover both mechanisms; with either fix reverted the
// corresponding case reports torn pages within seconds on a 4-core box.

#include <clio_cte/core/core_client.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/clio_runtime.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::run::u32;
using clio::run::u64;

namespace {

std::string TestDataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

constexpr u64 kPageBytes = 1ull << 20;  // 1 MiB, several bdev extents
constexpr u32 kBlobs = 8;
constexpr u32 kWriters = 4;
constexpr u32 kReaders = 8;
constexpr double kSeconds = 6.0;  // per case

class TornReadFixture {
 public:
  std::string config_path_;
  TornReadFixture() {
    config_path_ = TestDataDir() + "/torn_read_config.yaml";
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
  ~TornReadFixture() { Cleanup(); }
  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }
  void CreateConfigFile() {
    std::ofstream f(config_path_);
    REQUIRE(f.is_open());
    f << R"(
runtime:
  num_threads: 4
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
      - path: "ram::torn_read_dram"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
  }
};

/** The word at index i of blob b at generation g. */
inline u64 Word(u64 b, u64 g, u64 i) {
  u64 x = (b + 1) * 0x9E3779B97F4A7C15ull ^ (g + 1) * 0xC2B2AE3D27D4EB4Full ^ i;
  x ^= x >> 31; x *= 0x7FB5D329728EA185ull; x ^= x >> 27;
  x *= 0x81DADEF4BC2DD44Dull; x ^= x >> 33;
  return x;
}

void FillPage(char *buf, u64 b, u64 g) {
  u64 *w = reinterpret_cast<u64 *>(buf);
  const u64 n = kPageBytes / sizeof(u64);
  w[0] = b; w[1] = g;
  for (u64 i = 2; i < n; ++i) w[i] = Word(b, g, i);
}

/** @return -1 when the page is one whole generation of blob b, else the first
 *  word that disagrees with the page's own header. */
long long VerifyPage(const char *buf, u64 b) {
  const u64 *w = reinterpret_cast<const u64 *>(buf);
  const u64 n = kPageBytes / sizeof(u64);
  if (w[0] != b) return 0;
  const u64 g = w[1];
  for (u64 i = 2; i < n; ++i) if (w[i] != Word(b, g, i)) return (long long)i;
  return -1;
}

std::string Name(u32 b) { return "torn_" + std::to_string(b); }

clio::cte::core::Context Ctx(bool generational, u64 gen) {
  clio::cte::core::Context c;
  if (generational) {
    c.op_flags_ |= clio::cte::core::Context::kGenerational;
    c.generation_ = gen;
  }
  return c;
}

/**
 * Overwrite K blobs from W threads while R threads read and verify them.
 * @param generational  true: generational gets (runtime path); false: plain
 *                      gets (zero-IPC fast path from this process)
 * @return the number of torn pages observed
 */
u64 RunCase(bool generational) {
  auto *cte = CLIO_CTE_CLIENT;
  auto *ipc = CLIO_IPC;
  auto tag_task = cte->AsyncGetOrCreateTag(generational ? "torn_gen" : "torn_plain");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  const clio::cte::core::TagId tag = tag_task->tag_id_;

  // Seed generation 1 of every blob so readers always find something.
  {
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kPageBytes);
    for (u32 b = 0; b < kBlobs; ++b) {
      FillPage(buf.ptr_, b, 1);
      auto t = cte->AsyncPutBlob(tag, Name(b), 0, kPageBytes,
                                 buf.shm_.template Cast<void>(), 1.0f,
                                 Ctx(generational, 1));
      t.Wait();
      REQUIRE(t->GetReturnCode() == 0);
    }
    ipc->FreeBuffer(buf);
  }

  std::atomic<bool> stop{false};
  std::atomic<u64> puts{0}, gets{0}, torn{0}, get_errors{0}, put_errors{0};
  std::atomic<u64> gen[kBlobs];
  for (auto &g : gen) g.store(1);

  auto writer = [&](u32 w) {
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kPageBytes);
    u32 b = w % kBlobs;
    while (!stop.load()) {
      const u64 g = gen[b].fetch_add(1) + 1;
      FillPage(buf.ptr_, b, g);
      auto t = cte->AsyncPutBlob(tag, Name(b), 0, kPageBytes,
                                 buf.shm_.template Cast<void>(), 1.0f,
                                 Ctx(generational, g));
      t.Wait();
      if (t->GetReturnCode() != 0) put_errors++;
      puts++;
      b = (b + kWriters) % kBlobs;
    }
    ipc->FreeBuffer(buf);
  };
  auto reader = [&](u32 r) {
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(kPageBytes);
    u32 b = r % kBlobs;
    while (!stop.load()) {
      std::memset(buf.ptr_, 0, kPageBytes);
      // A generational get demands gen 1: every version satisfies it, so the
      // get never waits and the read overlaps the writers as much as possible.
      auto t = cte->AsyncGetBlob(tag, Name(b), 0, kPageBytes, 0u,
                                 buf.shm_.template Cast<void>(),
                                 clio::run::PoolQuery::Dynamic(),
                                 Ctx(generational, 1));
      t.Wait();
      if (t->GetReturnCode() != 0) {
        get_errors++;
      } else {
        const long long bad = VerifyPage(buf.ptr_, b);
        if (bad >= 0) {
          const u64 *w = reinterpret_cast<const u64 *>(buf.ptr_);
          if (torn.fetch_add(1) < 4) {
            HLOG(kError,
                 "[torn_read] {} get of blob {} returned a TORN page: header "
                 "gen {} but word {} disagrees",
                 generational ? "generational" : "plain", b, w[1], bad);
          }
        }
      }
      gets++;
      b = (b + 1) % kBlobs;
    }
    ipc->FreeBuffer(buf);
  };

  std::vector<std::thread> ts;
  for (u32 w = 0; w < kWriters; ++w) ts.emplace_back(writer, w);
  for (u32 r = 0; r < kReaders; ++r) ts.emplace_back(reader, r);
  std::this_thread::sleep_for(std::chrono::duration<double>(kSeconds));
  stop.store(true);
  for (auto &t : ts) t.join();

  HLOG(kInfo, "[torn_read] {} case: {} puts, {} gets, {} torn, {} get errors, {} put errors",
       generational ? "generational" : "plain", puts.load(), gets.load(),
       torn.load(), get_errors.load(), put_errors.load());
  REQUIRE(put_errors.load() == 0);
  REQUIRE(get_errors.load() == 0);
  REQUIRE(puts.load() > 0);
  REQUIRE(gets.load() > 0);
  return torn.load();
}

}  // namespace

TEST_CASE("TornRead - generational gets never see a mixture of two versions",
          "[torn_read][cte][regression]") {
  static TornReadFixture fixture;
  REQUIRE(RunCase(/*generational=*/true) == 0);
}

TEST_CASE("TornRead - plain gets (zero-IPC fast path) never see a mixture",
          "[torn_read][cte][regression]") {
  static TornReadFixture fixture;
  REQUIRE(RunCase(/*generational=*/false) == 0);
}

SIMPLE_TEST_MAIN()
