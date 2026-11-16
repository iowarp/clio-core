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
 * Unit tests for the core_shm_cache_writethru chimod and the core client's cache
 * path (context-transfer-engine/core_shm_cache_writethru/DESIGN.md).
 *
 * Chain under test: core_shm_cache_writethru (568.0) -> core (512.0). `io` is a
 * core client bound to the cache pool (SHM path); `core` reads the chain
 * directly to check what actually landed.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_client.h>

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

/** Create a cache pool over the core (data_capacity > 0: extent mode, its
 *  bytes in granules leased from the core's RAM bdev on this node). */
static void CreateCache(const clio::run::PoolId &id, const std::string &name,
                        clio::run::u64 capacity, bool md_only = false,
                        clio::run::u64 data_capacity = 0) {
  clio::cte::shm_wtc::Client admin;
  clio::cte::shm_wtc::WtcConfig params;
  params.md_only_ = md_only;
  if (data_capacity > 0) {
    params.data_pool_ = clio::run::PoolId(512, 0);  // this node's RAM tier
    params.data_capacity_ = data_capacity;
  }
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

static const clio::run::PoolId kPool(568, 0);

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

  SECTION("reads inside written bytes of a partial blob hit") {
    // The blob exists below (the cache did not create it), so the cached
    // copy is never complete; the bytes written through the cache are still
    // the blob's current bytes and serve reads of exactly that range.
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    TagId tag = MakeTag(core, "wtc_valid_range");
    const std::string base(128 * 1024, 'x');
    core.AsyncPutBlob(tag, "v", 0, base.size(), base.data()).Wait();
    const std::string a(8192, 'A');
    io.AsyncPutBlob(tag, "v", 4096, a.size(), a.data()).Wait();

    std::string back(4096, '\0');
    const clio::run::u64 hits0 = st->get_hits_.load();
    const clio::run::u64 fills0 = st->fills_.load();
    io.AsyncGetBlob(tag, "v", 8192, back.size(), 0, back.data()).Wait();
    REQUIRE(back == std::string(4096, 'A'));
    REQUIRE(st->get_hits_.load() == hits0 + 1);  // served from the segment
    REQUIRE(st->fills_.load() == fills0);         // no trip to the chain

    // A read touching bytes the cache never saw is the chain's (not zeros).
    std::string span(8192, '\0');
    auto g = io.AsyncGetBlob(tag, "v", 0, span.size(), 0, span.data());
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(span == std::string(4096, 'x') + std::string(4096, 'A'));

    // Another writer's invalidation: the written range no longer serves.
    const std::string c(8192, 'C');
    io.AsyncPutBlob(tag, "v", 4096, c.size(), c.data()).Wait();
    REQUIRE(WaitFlushed(io));
    io.Wtc()->Invalidate(tag.major_, tag.minor_, "v", 1);
    core.AsyncPutBlob(tag, "v", 4096, a.size(), a.data()).Wait();
    io.AsyncGetBlob(tag, "v", 8192, back.size(), 0, back.data()).Wait();
    REQUIRE(back == std::string(4096, 'A'));  // the chain's newer bytes
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
  const clio::run::PoolId pool(569, 0);
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

TEST_CASE("ShmWtc - overwrites flush once and the last write wins",
          "[cte][shm_wtc][dedup]") {
  WtcFixture fx;
  clio::cte::core::Client io(kPool);
  REQUIRE(io.HasWtc());
  const wtc::Stats *st = io.WtcStats();
  TagId tag = MakeTag(io, "wtc_dedup");
  REQUIRE(WaitFlushed(io));
  const clio::run::u64 skipped0 = st->flush_skipped_.load();

  // Many overwrites of a few blobs, faster than one flush each: ranges an
  // earlier flush copy already carried are retired without a resend.
  const int kBlobs = 4;
  const int kRounds = 400;
  std::vector<std::string> last(kBlobs);
  for (int r = 0; r < kRounds; ++r) {
    for (int b = 0; b < kBlobs; ++b) {
      last[b] = Payload(64 << 10, static_cast<char>('a' + (r + b) % 26));
      auto p = io.AsyncPutBlob(tag, "d" + std::to_string(b), 0,
                               last[b].size(), last[b].data());
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
    }
  }
  // A partial overwrite after all that must not be skipped.
  std::string tail = Payload(4096, 'Z');
  {
    auto p = io.AsyncPutBlob(tag, "d0", 8192, tail.size(), tail.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
    std::memcpy(&last[0][8192], tail.data(), tail.size());
  }
  REQUIRE(WaitFlushed(io));
  INFO("skipped " << st->flush_skipped_.load() - skipped0);
  for (int b = 0; b < kBlobs; ++b) {
    REQUIRE(CoreRead(tag, "d" + std::to_string(b), 0, last[b].size()) ==
            last[b]);
  }
  REQUIRE(st->flush_skipped_.load() > skipped0);
}

/** Size of a blob through a client (served from SHM when cached). */
static clio::run::u64 SizeOf(clio::cte::core::Client &io, const TagId &tag,
                             const std::string &name) {
  auto f = io.AsyncGetBlobSize(tag, name);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->size_;
}

TEST_CASE("ShmWtc - md_only caches blob sizes and drops them on change",
          "[cte][shm_wtc][md_only]") {
  WtcFixture fx;
  const clio::run::PoolId pool(571, 0);
  CreateCache(pool, "shm_wtc_md", 32ull << 20, /*md_only=*/true);
  clio::cte::core::Client io(pool);
  REQUIRE(io.HasWtc());
  const wtc::Stats *st = io.WtcStats();
  TagId tag = MakeTag(io, "wtc_md");

  // Bytes are not cached: a put takes the task path and lands in the core.
  const std::string v1 = Payload(1000, 'a');
  auto p = io.AsyncPutBlob(tag, "m", 0, v1.size(), v1.data());
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);
  REQUIRE(st->puts_.load() == 0);
  REQUIRE(CoreRead(tag, "m", 0, v1.size()) == v1);

  // First size query fetches and caches; the second is served from SHM.
  REQUIRE(SizeOf(io, tag, "m") == 1000);
  REQUIRE(st->md_fills_.load() >= 1);
  const clio::run::u64 fills = st->md_fills_.load();
  REQUIRE(SizeOf(io, tag, "m") == 1000);
  REQUIRE(st->md_fills_.load() == fills);

  // A growing write, then a truncate: the cached size never goes stale.
  const std::string v2 = Payload(3000, 'b');
  p = io.AsyncPutBlob(tag, "m", 0, v2.size(), v2.data());
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);
  REQUIRE(SizeOf(io, tag, "m") == 3000);
  auto t = io.AsyncTruncateBlob(tag, "m", 700);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  REQUIRE(SizeOf(io, tag, "m") == 700);
  std::string back(700, '\0');
  auto g = io.AsyncGetBlob(tag, "m", 0, back.size(), 0, back.data());
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  REQUIRE(back == v2.substr(0, 700));
}

/** Cluster-wide tag size through a client (Broadcast). */
static clio::run::u64 TagSizeOf(clio::cte::core::Client &io, const TagId &tag) {
  auto f = io.AsyncGetTagSize(tag, clio::run::PoolQuery::Broadcast(0.0f));
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->tag_size_;
}

/** Tag attributes and open-by-name, served from SHM and never stale. */
static void TagMetaScenario(clio::cte::core::Client &io, const char *tname) {
  const wtc::Stats *st = io.WtcStats();
  TagId tag = MakeTag(io, tname);
  auto put = [&](const std::string &name, size_t n, char c) {
    const std::string v = Payload(n, c);
    auto p = io.AsyncPutBlob(tag, name, 0, v.size(), v.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  };
  put("a", 1000, 'a');
  REQUIRE(TagSizeOf(io, tag) == 1000);  // read-your-writes (absorbed or not)
  const clio::run::u64 hits = st->tag_hits_.load();
  REQUIRE(TagSizeOf(io, tag) == 1000);
  REQUIRE(st->tag_hits_.load() > hits);  // served from the segment
  put("b", 500, 'b');
  REQUIRE(TagSizeOf(io, tag) == 1500);   // a local write voids the cache
  auto t = io.AsyncTruncateBlob(tag, "a", 100);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  REQUIRE(TagSizeOf(io, tag) == 600);
  // Open by name: the second open is answered from the segment.
  auto o1 = io.AsyncGetOrCreateTag(tname);
  o1.Wait();
  REQUIRE(o1->GetReturnCode() == 0);
  REQUIRE(o1->tag_id_ == tag);
  REQUIRE(TagSizeOf(io, tag) == 600);
  const clio::run::u64 hits2 = st->tag_hits_.load();
  auto o2 = io.AsyncGetOrCreateTag(tname);
  o2.Wait();
  REQUIRE(o2->GetReturnCode() == 0);
  REQUIRE(o2->tag_id_ == tag);
  REQUIRE(o2->tag_size_ == 600);
  REQUIRE(st->tag_hits_.load() > hits2);
  // Delete the tag: the name no longer maps to the old id.
  auto d = io.AsyncDelTag(tag);
  d.Wait();
  auto o3 = io.AsyncGetOrCreateTag(tname);
  o3.Wait();
  REQUIRE(o3->GetReturnCode() == 0);
  REQUIRE(o3->created_ == 1);
  REQUIRE(TagSizeOf(io, o3->tag_id_) == 0);
}

TEST_CASE("ShmWtc - tag attributes and opens are cached and exact (data)",
          "[cte][shm_wtc][tagmeta]") {
  WtcFixture fx;
  clio::cte::core::Client io(kPool);
  REQUIRE(io.HasWtc());
  TagMetaScenario(io, "wtc_tagmeta_data");
}

TEST_CASE("ShmWtc - tag attributes and opens are cached and exact (md_only)",
          "[cte][shm_wtc][tagmeta][md_only]") {
  WtcFixture fx;
  const clio::run::PoolId pool(572, 0);
  CreateCache(pool, "shm_wtc_md2", 32ull << 20, /*md_only=*/true);
  clio::cte::core::Client io(pool);
  REQUIRE(io.HasWtc());
  TagMetaScenario(io, "wtc_tagmeta_md");
}

TEST_CASE("ShmWtc - truncate through the cache drops cached bytes",
          "[cte][shm_wtc][truncate]") {
  WtcFixture fx;
  clio::cte::core::Client io(kPool);
  REQUIRE(io.HasWtc());
  TagId tag = MakeTag(io, "wtc_trunc");
  const std::string v = Payload(8192, 'q');
  auto p = io.AsyncPutBlob(tag, "t", 0, v.size(), v.data());
  p.Wait();
  REQUIRE(p->GetReturnCode() == 0);
  REQUIRE(SizeOf(io, tag, "t") == 8192);  // cached with its bytes
  auto t = io.AsyncTruncateBlob(tag, "t", 100);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  REQUIRE(SizeOf(io, tag, "t") == 100);
  std::string back(100, '\0');
  auto g = io.AsyncGetBlob(tag, "t", 0, back.size(), 0, back.data());
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  REQUIRE(back == v.substr(0, 100));
}

TEST_CASE("ShmWtc - large puts displace a cache full of small buffers",
          "[cte][shm_wtc][evict][fragment]") {
  WtcFixture fx;
  const clio::run::PoolId pool(570, 0);
  CreateCache(pool, "shm_wtc_frag", 8ull << 20);  // 8 MB segment
  clio::cte::core::Client io(pool);
  REQUIRE(io.HasWtc());
  const wtc::Stats *st = io.WtcStats();
  TagId tag = MakeTag(io, "wtc_frag");

  // Fill the segment with clean 4 KB buffers: freeing any one of them never
  // leaves room for a 1 MB buffer, since the allocator does not merge on
  // free.
  const std::string small = Payload(4096, 's');
  for (int i = 0; i < 4000; ++i) {
    auto p = io.AsyncPutBlob(tag, "s" + std::to_string(i), 0, small.size(),
                             small.data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  }
  REQUIRE(WaitFlushed(io));

  // Every large put must land in the cache (puts_ counts cached writes
  // only; a write-around bypasses it), and quickly.
  const clio::run::u64 puts0 = st->puts_.load();
  std::vector<std::string> vals;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 4; ++i) {
    vals.push_back(Payload(1 << 20, static_cast<char>('a' + i)));
    auto p = io.AsyncPutBlob(tag, "L" + std::to_string(i), 0,
                             vals.back().size(), vals.back().data());
    p.Wait();
    REQUIRE(p->GetReturnCode() == 0);
  }
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  INFO("large puts took " << ms << " ms");
  REQUIRE(st->puts_.load() - puts0 == 4);
  REQUIRE(ms < 2000.0);
  REQUIRE(WaitFlushed(io));
  for (int i = 0; i < 4; ++i) {
    std::string back(1 << 20, '\0');
    auto g = io.AsyncGetBlob(tag, "L" + std::to_string(i), 0, back.size(), 0,
                             back.data());
    g.Wait();
    REQUIRE(g->GetReturnCode() == 0);
    REQUIRE(back == vals[i]);
    REQUIRE(CoreRead(tag, "L" + std::to_string(i), 0, back.size()) ==
            vals[i]);
  }
}

TEST_CASE("ShmWtc - extent mode keeps bytes in leased RAM-bdev granules",
          "[cte][shm_wtc][extents]") {
  WtcFixture fx;
  const clio::run::PoolId kExtPool(575, 0);
  // 8 MiB of metadata segment, 16 MiB of leased RAM-bdev extents.
  CreateCache(kExtPool, "shm_wtc_extents", 8ull << 20, false, 16ull << 20);
  clio::cte::core::Client io(kExtPool);
  REQUIRE(io.HasWtc());
  const wtc::Stats *st = io.WtcStats();
  REQUIRE(st != nullptr);

  SECTION("sizes that are not powers of two round trip and flush") {
    TagId tag = MakeTag(io, "ext_sizes");
    const std::vector<size_t> sizes = {1, 4096, 4097, 100 * 1024,
                                       (1u << 20) + 1};
    for (size_t i = 0; i < sizes.size(); ++i) {
      const std::string name = "s" + std::to_string(i);
      const std::string v = Payload(sizes[i], static_cast<char>('a' + i));
      auto put = io.AsyncPutBlob(tag, name, 0, v.size(), v.data());
      put.Wait();
      REQUIRE(put->GetReturnCode() == 0);
      std::string back(v.size(), '\0');
      const clio::run::u64 hits0 = st->get_hits_.load();
      auto get = io.AsyncGetBlob(tag, name, 0, back.size(), 0, back.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(back == v);
      REQUIRE(st->get_hits_.load() == hits0 + 1);  // served from the bdev
    }
    REQUIRE(WaitFlushed(io));
    for (size_t i = 0; i < sizes.size(); ++i) {
      const std::string v = Payload(sizes[i], static_cast<char>('a' + i));
      REQUIRE(CoreRead(tag, "s" + std::to_string(i), 0, v.size()) == v);
    }
  }

  SECTION("a blob grows by appending runs; its bytes never move") {
    TagId tag = MakeTag(io, "ext_grow");
    std::string want;
    for (int k = 0; k < 6; ++k) {  // 6 appends of 300 KiB at the tail
      const std::string piece = Payload(300 * 1024, static_cast<char>('A' + k));
      auto put = io.AsyncPutBlob(tag, "g", want.size(), piece.size(),
                                 piece.data());
      put.Wait();
      REQUIRE(put->GetReturnCode() == 0);
      want += piece;
    }
    std::string back(want.size(), '\0');
    io.AsyncGetBlob(tag, "g", 0, back.size(), 0, back.data()).Wait();
    REQUIRE(back == want);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "g", 0, want.size()) == want);
  }

  SECTION("a working set larger than the lease evicts and stays correct") {
    TagId tag = MakeTag(io, "ext_evict");
    const int kBlobs = 40;  // 40 MiB through a 16 MiB lease
    for (int k = 0; k < kBlobs; ++k) {
      const std::string v = Payload(1u << 20, static_cast<char>(k));
      auto put = io.AsyncPutBlob(tag, "e" + std::to_string(k), 0, v.size(),
                                 v.data());
      put.Wait();
      REQUIRE(put->GetReturnCode() == 0);
    }
    REQUIRE(WaitFlushed(io));
    for (int k = 0; k < kBlobs; ++k) {
      const std::string v = Payload(1u << 20, static_cast<char>(k));
      std::string back(v.size(), '\0');
      auto get = io.AsyncGetBlob(tag, "e" + std::to_string(k), 0, back.size(),
                                 0, back.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(back == v);
    }
    REQUIRE(st->evictions_.load() > 0);
  }

  SECTION("large puts into a lease fragmented by small blobs stay cached") {
    TagId tag = MakeTag(io, "ext_frag");
    // Fill the whole lease with clean single-granule blobs: evicting any
    // one frees one scattered granule, never a run a 1 MiB blob can use.
    const std::string small = Payload(4096, 's');
    for (int i = 0; i < 4200; ++i) {
      auto p = io.AsyncPutBlob(tag, "s" + std::to_string(i), 0, small.size(),
                               small.data());
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
    }
    REQUIRE(WaitFlushed(io));
    const clio::run::u64 puts0 = st->puts_.load();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; ++i) {
      const std::string v = Payload(1 << 20, static_cast<char>('a' + i));
      auto p = io.AsyncPutBlob(tag, "L" + std::to_string(i), 0, v.size(),
                               v.data());
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    INFO("large puts took " << ms << " ms");
    REQUIRE(st->puts_.load() - puts0 == 4);  // cached, not written around
    REQUIRE(ms < 2000.0);
    REQUIRE(WaitFlushed(io));
    for (int i = 0; i < 4; ++i) {
      const std::string v = Payload(1 << 20, static_cast<char>('a' + i));
      REQUIRE(CoreRead(tag, "L" + std::to_string(i), 0, v.size()) == v);
    }
  }

  SECTION("a partial write into a blob that exists below") {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    TagId tag = MakeTag(core, "ext_partial");
    const std::string base(128 * 1024, 'x');
    core.AsyncPutBlob(tag, "p", 0, base.size(), base.data()).Wait();
    const std::string a(5000, 'A');
    io.AsyncPutBlob(tag, "p", 70000, a.size(), a.data()).Wait();
    std::string want = base;
    std::memcpy(&want[70000], a.data(), a.size());
    std::string back(want.size(), '\0');
    io.AsyncGetBlob(tag, "p", 0, back.size(), 0, back.data()).Wait();
    REQUIRE(back == want);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "p", 0, want.size()) == want);
  }
}

SIMPLE_TEST_MAIN()
