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
 * Unit tests for the shm_writethru_cache chimod and the core client's cache
 * path (context-transfer-engine/shm_writethru_cache/DESIGN.md).
 *
 * Chain under test: shm_writethru_cache (565.0) -> core (512.0). `io` is a
 * core client bound to the cache pool (SHM path); `core` reads the chain
 * directly to check what actually landed.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_client.h>

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
namespace wtc = clio::cte::core::wtc;
using clio::cte::core::TagId;

/** Test data directory (CLIO_TEST_DATA_DIR). */
static std::string DataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Brings up a runtime with a RAM core pool and the cache over it. */
class WtcFixture {
 public:
  std::string config_path_;
  std::string restart_log_path_;

  WtcFixture() {
    config_path_ = DataDir() + "/shm_wtc_config.yaml";
    restart_log_path_ = DataDir() + "/shm_wtc_restart.bin";
    Cleanup();
    WriteConfig();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_path_.c_str(), 1);
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
  }

  ~WtcFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(restart_log_path_)) fs::remove(restart_log_path_);
  }

  void WriteConfig() {
    std::ofstream f(config_path_);
    REQUIRE(f.is_open());
    f << R"(
runtime:
  num_threads: 4
  queue_depth: 1024

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
      - path: "ram::shm_wtc_dram"
        bdev_type: "ram"
        capacity_limit: "512MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
  }
};

/** Create a cache pool over the core. */
static void CreateCache(const clio::run::PoolId &id, const std::string &name,
                        clio::run::u64 capacity) {
  clio::cte::shm_wtc::Client admin;
  clio::cte::shm_wtc::WtcConfig params;
  params.next_pool_id_ = clio::cte::core::kCtePoolId;
  params.capacity_ = capacity;
  params.max_blobs_ = 4096;
  params.log_depth_ = 4096;
  auto create = admin.AsyncCreate(clio::run::PoolQuery::Local(), name, id,
                                  params);
  create.Wait();
  REQUIRE(create->GetReturnCode() == 0);
}

/** Deterministic payload. */
static std::string Payload(size_t n, char seed) {
  std::string v(n, '\0');
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<char>(seed + (i % 53));
  return v;
}

/** Wait (bounded) until the cache has flushed every range pushed so far. */
static bool WaitFlushed(clio::cte::core::Client &io) {
  wtc::View *view = io.Wtc();
  if (view == nullptr) return false;
  for (int i = 0; i < 2000; ++i) {
    const wtc::Root *root = view->GetRoot();
    if (root->flushed_.load() >= root->pushed_.load()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

/** Read a blob range straight from the core (bypassing the cache). */
static std::string CoreRead(const TagId &tag, const std::string &name,
                            clio::run::u64 off, clio::run::u64 size) {
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  std::string out(size, '\0');
  auto get = core.AsyncGetBlob(tag, name, off, size, 0, out.data());
  get.Wait();
  REQUIRE(get->GetReturnCode() == 0);
  return out;
}

/** Create a tag through a client. */
static TagId MakeTag(clio::cte::core::Client &c, const std::string &name) {
  auto t = c.AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

static const clio::run::PoolId kPool(565, 0);

TEST_CASE("ShmWtc - writes and reads complete in shared memory",
          "[cte][shm_wtc]") {
  WtcFixture fx;
  CreateCache(kPool, "shm_wtc_main", 64ull << 20);
  clio::cte::core::Client io(kPool);
  REQUIRE(io.HasWtc());
  const wtc::Stats *st = io.WtcStats();
  REQUIRE(st != nullptr);

  SECTION("put/get round trip without tasks, then write-through") {
    TagId tag = MakeTag(io, "wtc_roundtrip");
    const std::string val = Payload(256 * 1024, 'a');
    const clio::run::u64 puts0 = st->puts_.load();
    auto put = io.AsyncPutBlob(tag, "b0", 0, val.size(), val.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    REQUIRE(st->puts_.load() == puts0 + 1);

    // A brand-new blob of a brand-new tag is complete in the cache.
    std::string back(val.size(), '\0');
    const clio::run::u64 hits0 = st->get_hits_.load();
    auto get = io.AsyncGetBlob(tag, "b0", 0, back.size(), 0, back.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(back == val);
    REQUIRE(st->get_hits_.load() == hits0 + 1);

    auto sz = io.AsyncGetBlobSize(tag, "b0");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    REQUIRE(sz->size_ == val.size());

    // The flusher lands the bytes in the chain.
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "b0", 0, val.size()) == val);
  }

  SECTION("only the written ranges reach the chain") {
    // The blob already exists below with known bytes; the cache did not
    // create it, so its buffer is partial and holes must never be shipped.
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    TagId tag = MakeTag(core, "wtc_ranges");
    const std::string base(128 * 1024, 'x');
    auto p0 = core.AsyncPutBlob(tag, "r", 0, base.size(), base.data());
    p0.Wait();
    REQUIRE(p0->GetReturnCode() == 0);

    const std::string a(4096, 'A');
    const std::string b(4096, 'B');
    io.AsyncPutBlob(tag, "r", 0, a.size(), a.data()).Wait();
    io.AsyncPutBlob(tag, "r", 64 * 1024, b.size(), b.data()).Wait();
    REQUIRE(WaitFlushed(io));

    std::string want = base;
    std::memcpy(&want[0], a.data(), a.size());
    std::memcpy(&want[64 * 1024], b.data(), b.size());
    REQUIRE(CoreRead(tag, "r", 0, want.size()) == want);

    // Reads of a partially cached blob go to the chain (and fill).
    std::string back(want.size(), '\0');
    auto g = io.AsyncGetBlob(tag, "r", 0, back.size(), 0, back.data());
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(back == want);
  }

  SECTION("small writes coalesce into few batches") {
    TagId tag = MakeTag(io, "wtc_coalesce");
    std::string want(128 * 512, '\0');
    REQUIRE(WaitFlushed(io));
    const clio::run::u64 batches0 = st->flush_batches_.load();
    const clio::run::u64 ranges0 = st->flush_ranges_.load();
    for (int i = 0; i < 128; ++i) {
      std::string piece(512, static_cast<char>('a' + (i % 26)));
      std::memcpy(&want[i * 512], piece.data(), piece.size());
      auto p = io.AsyncPutBlob(tag, "seq", i * 512, piece.size(), piece.data());
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
    }
    REQUIRE(WaitFlushed(io));
    REQUIRE(st->flush_ranges_.load() - ranges0 == 128);
    // Many ranges per batch: far fewer batches than writes.
    REQUIRE(st->flush_batches_.load() - batches0 < 64);
    REQUIRE(CoreRead(tag, "seq", 0, want.size()) == want);
  }

  SECTION("read miss fills from the chain, then hits") {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    TagId tag = MakeTag(core, "wtc_fill");
    const std::string val = Payload(96 * 1024, 'f');
    auto p = core.AsyncPutBlob(tag, "cold", 0, val.size(), val.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);

    const clio::run::u64 fills0 = st->fills_.load();
    std::string back(val.size(), '\0');
    auto g1 = io.AsyncGetBlob(tag, "cold", 0, back.size(), 0, back.data());
    g1.Wait();
    REQUIRE(g1->GetReturnCode() == 0);
    REQUIRE(back == val);
    REQUIRE(st->fills_.load() == fills0 + 1);

    const clio::run::u64 hits0 = st->get_hits_.load();
    std::string back2(4096, '\0');
    auto g2 = io.AsyncGetBlob(tag, "cold", 8192, back2.size(), 0, back2.data());
    g2.Wait();
    REQUIRE(g2->GetReturnCode() == 0);
    REQUIRE(back2 == val.substr(8192, 4096));
    REQUIRE(st->get_hits_.load() == hits0 + 1);
  }

  SECTION("a non-plain put takes the task path and stays ordered") {
    TagId tag = MakeTag(io, "wtc_taskpath");
    const std::string v1(8192, '1');
    const std::string v2(8192, '2');
    io.AsyncPutBlob(tag, "t", 0, v1.size(), v1.data()).Wait();
    // An explicit score is not expressible in the cache: task path, after
    // the buffered write lands.
    auto p2 = io.AsyncPutBlob(tag, "t", 0, v2.size(), v2.data(), 0.75f);
    p2.Wait();
    REQUIRE(p2->GetReturnCode() == 0);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "t", 0, v2.size()) == v2);
    std::string back(v2.size(), '\0');
    auto g = io.AsyncGetBlob(tag, "t", 0, back.size(), 0, back.data());
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(back == v2);
  }

  SECTION("DelBlob removes the cached copy and the chain's") {
    TagId tag = MakeTag(io, "wtc_del");
    const std::string v(4096, 'd');
    io.AsyncPutBlob(tag, "gone", 0, v.size(), v.data()).Wait();
    auto d = io.AsyncDelBlob(tag, "gone");
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    auto sz = io.AsyncGetBlobSize(tag, "gone");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() != 0);
  }

  SECTION("concurrent writers and readers") {
    TagId tag = MakeTag(io, "wtc_mt");
    std::atomic<int> bad{0};
    std::vector<std::thread> ths;
    for (int t = 0; t < 4; ++t) {
      ths.emplace_back([&io, &bad, tag, t] {
        for (int i = 0; i < 64; ++i) {
          const std::string name = "mt_" + std::to_string(t) + "_" +
                                   std::to_string(i);
          const std::string v = Payload(16 * 1024 + i, static_cast<char>('a' + t));
          auto p = io.AsyncPutBlob(tag, name, 0, v.size(), v.data());
          p.Wait();
          std::string back(v.size(), '\0');
          auto g = io.AsyncGetBlob(tag, name, 0, back.size(), 0, back.data());
          g.Wait();
          if (p->GetReturnCode() != 0 || g->GetReturnCode() != 0 || back != v) {
            ++bad;
          }
        }
      });
    }
    for (auto &th : ths) th.join();
    REQUIRE(bad.load() == 0);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "mt_3_63", 0, 16 * 1024 + 63) ==
            Payload(16 * 1024 + 63, 'd'));
  }
}

TEST_CASE("ShmWtc - eviction keeps a small cache writable",
          "[cte][shm_wtc][evict]") {
  WtcFixture fx;
  const clio::run::PoolId pool(566, 0);
  CreateCache(pool, "shm_wtc_small", 8ull << 20);  // 8 MB segment
  clio::cte::core::Client io(pool);
  REQUIRE(io.HasWtc());
  const wtc::Stats *st = io.WtcStats();
  TagId tag = MakeTag(io, "wtc_evict");

  // 48 MB through an 8 MB cache: only possible if clean buffers are evicted.
  std::vector<std::string> vals;
  for (int i = 0; i < 48; ++i) {
    vals.push_back(Payload(1 << 20, static_cast<char>('A' + (i % 26))));
    auto p = io.AsyncPutBlob(tag, "e" + std::to_string(i), 0, vals.back().size(),
                             vals.back().data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  }
  REQUIRE(WaitFlushed(io));
  REQUIRE(st->evictions_.load() > 0);
  for (int i = 0; i < 48; ++i) {
    std::string back(1 << 20, '\0');
    auto g = io.AsyncGetBlob(tag, "e" + std::to_string(i), 0, back.size(), 0,
                             back.data());
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(back == vals[i]);
  }
}

SIMPLE_TEST_MAIN()
