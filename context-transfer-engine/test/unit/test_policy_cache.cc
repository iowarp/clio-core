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
 * Policy caches over a volatile and a persistent core.
 *
 * One runtime composes two DISCONNECTED cores -- persistent 512.0 (file tier,
 * WAL on) and volatile 540.0 (a small RAM tier, given a metadata_log_path it
 * must ignore) -- and the test creates three caches over them:
 *   566.0 write-through, 567.0 write-back (fallback on, no periodic flush),
 *   568.0 write-back with fallback_to_persistent: false.
 * Each section uses its own tag, and reads the cores directly to see where
 * the bytes are.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_config.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/write_back_cache/write_back_cache_client.h>
#include <clio_cte/write_thru_cache/write_thru_cache_client.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::TagId;

static std::string TestDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

static const clio::run::PoolId kPersistentPool(512, 0);
static const clio::run::PoolId kVolatilePool(540, 0);
static const clio::run::PoolId kThruPool(566, 0);
static const clio::run::PoolId kBackPool(567, 0);
static const clio::run::PoolId kStrictPool(568, 0);
// Durability cores (majors spaced: bdev ids are core major + device index).
/** Unstated durability, persistent bdev only: inferred persistent. */
static const clio::run::PoolId kInferPersPool(580, 0);
/** Unstated durability, persistent + volatile bdevs: inferred volatile. */
static const clio::run::PoolId kInferVolPool(585, 0);
/** durability: volatile stated over a persistent bdev; switched at runtime. */
static const clio::run::PoolId kSwitchPool(590, 0);
/** Volatile RAM tier: small, so eviction and fallback are reachable. */
static constexpr clio::run::u64 kVolatileBytes = 8ULL << 20;
static constexpr clio::run::u64 kMiB = 1ULL << 20;

/** Paths the fixture owns. */
struct Paths {
  std::string dir = TestDir();
  std::string config = dir + "/policy_cache_config.yaml";
  std::string file_tier = dir + "/policy_cache_file.dat";
  std::string pers_log = dir + "/policy_cache_pers_log";
  std::string vol_log = dir + "/policy_cache_vol_log";
  std::string restart = dir + "/policy_cache_restart.bin";
  std::string dur_file = dir + "/policy_cache_dur_file.dat";
  std::string dur_mixed_file = dir + "/policy_cache_dur_mixed.dat";
  std::string dur_switch_file = dir + "/policy_cache_dur_switch.dat";
  std::string dur_extra_file = dir + "/policy_cache_dur_extra.dat";
  std::string dur_pers_log = dir + "/policy_cache_dur_pers_log";
  std::string dur_mixed_log = dir + "/policy_cache_dur_mixed_log";
  std::string dur_switch_log = dir + "/policy_cache_dur_switch_log";
};

/** Remove every file under the test dir whose name starts with `prefix`. */
static void RemovePrefixed(const std::string &dir, const std::string &prefix) {
  std::error_code ec;
  for (const auto &e : fs::directory_iterator(dir, ec)) {
    if (e.path().filename().string().rfind(prefix, 0) == 0) {
      fs::remove_all(e.path(), ec);
    }
  }
}

/** Count files under `dir` whose names start with `prefix`. */
static size_t CountPrefixed(const std::string &dir, const std::string &prefix) {
  size_t n = 0;
  std::error_code ec;
  for (const auto &e : fs::directory_iterator(dir, ec)) {
    if (e.path().filename().string().rfind(prefix, 0) == 0) ++n;
  }
  return n;
}

/** Write the compose config: the two cores, nothing else. */
static void WriteConfig(const Paths &p) {
  std::ofstream f(p.config);
  REQUIRE(f.is_open());
  f << R"(
runtime:
  num_threads: 2
  queue_depth: 1024

compose:
  - mod_name: clio_cte_core
    pool_name: cte_persistent
    pool_query: local
    pool_id: 512.0
    durability: persistent
    targets:
      neighborhood: 1
    storage:
      - path: ")" << p.file_tier << R"("
        bdev_type: "file"
        capacity_limit: "256MB"
        score: 0.2
        persistence_level: "temporary"
    performance:
      metadata_log_path: ")" << p.pers_log << R"("

  - mod_name: clio_cte_core
    pool_name: cte_volatile
    pool_query: local
    pool_id: 540.0
    targets:
      neighborhood: 1
    storage:
      - path: "ram::policy_cache_vram"
        bdev_type: "ram"
        capacity_limit: "8MB"
        score: 1.0
    performance:
      metadata_log_path: ")" << p.vol_log << R"("

  - mod_name: clio_cte_core
    pool_name: cte_dur_inferred_persistent
    pool_query: local
    pool_id: 580.0
    targets:
      neighborhood: 1
    storage:
      - path: ")" << p.dur_file << R"("
        bdev_type: "file"
        capacity_limit: "16MB"
        persistence_level: "temporary"
    performance:
      metadata_log_path: ")" << p.dur_pers_log << R"("

  - mod_name: clio_cte_core
    pool_name: cte_dur_inferred_volatile
    pool_query: local
    pool_id: 585.0
    targets:
      neighborhood: 1
    storage:
      - path: ")" << p.dur_mixed_file << R"("
        bdev_type: "file"
        capacity_limit: "16MB"
        persistence_level: "temporary"
      - path: "ram::policy_cache_dur_mixed_ram"
        bdev_type: "ram"
        capacity_limit: "8MB"
    performance:
      metadata_log_path: ")" << p.dur_mixed_log << R"("

  - mod_name: clio_cte_core
    pool_name: cte_dur_switch
    pool_query: local
    pool_id: 590.0
    durability: volatile
    targets:
      neighborhood: 1
    storage:
      - path: ")" << p.dur_switch_file << R"("
        bdev_type: "file"
        capacity_limit: "16MB"
        persistence_level: "temporary"
    performance:
      metadata_log_path: ")" << p.dur_switch_log << R"("
)";
}

/** Runtime + cache pools, created once for the whole binary. */
class PolicyCacheFixture {
 public:
  Paths p_;

  PolicyCacheFixture() {
    RemovePrefixed(p_.dir, "policy_cache_");
    WriteConfig(p_);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", p_.config.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", p_.restart.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    CreateCaches();
  }

  void CreateCaches() {
    {
      clio::cte::write_thru_cache::Client c;
      clio::cte::write_thru_cache::WriteThruCacheConfig params;
      params.volatile_pool_ = kVolatilePool;
      params.persistent_pool_ = kPersistentPool;
      auto f = c.AsyncCreateCache(clio::run::PoolQuery::Local(),
                                  "policy_thru", kThruPool, params);
      f.Wait();
      REQUIRE(f->GetReturnCode() == 0);
    }
    for (int strict = 0; strict < 2; ++strict) {
      clio::cte::write_back_cache::Client c;
      clio::cte::write_back_cache::WriteBackCacheConfig params;
      params.volatile_pool_ = kVolatilePool;
      params.persistent_pool_ = kPersistentPool;
      params.flush_period_ms_ = 0;  // flushes only when the test asks
      params.fallback_to_persistent_ = strict == 0;
      auto f = c.AsyncCreateCache(
          clio::run::PoolQuery::Local(), strict ? "policy_strict" : "policy_back",
          strict ? kStrictPool : kBackPool, params);
      f.Wait();
      REQUIRE(f->GetReturnCode() == 0);
    }
  }
};

static PolicyCacheFixture *Fixture() {
  static PolicyCacheFixture *f = new PolicyCacheFixture();
  return f;
}

/** A tag created on the persistent core (the tag authority). */
static TagId MakeTag(const std::string &name) {
  clio::cte::core::Client pers(kPersistentPool);
  auto f = pers.AsyncGetOrCreateTag(name);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->tag_id_;
}

/** Deterministic payload; `seed` changes every byte. */
static std::string Payload(clio::run::u64 size, char seed) {
  std::string v(size, '\0');
  for (clio::run::u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>(seed + (i * 7) % 61);
  }
  return v;
}

static clio::run::u32 Put(const clio::run::PoolId &pool, const TagId &tag,
                          const std::string &name, clio::run::u64 off,
                          const std::string &data) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncPutBlob(tag, name, off, data.size(), data.data());
  f.Wait();
  return f.get() == nullptr ? 1u : f->GetReturnCode();
}

/** Read [off, off+size); empty string when the read fails. */
static std::string Get(const clio::run::PoolId &pool, const TagId &tag,
                       const std::string &name, clio::run::u64 off,
                       clio::run::u64 size) {
  clio::cte::core::Client c(pool);
  std::string out(size, '\0');
  auto f = c.AsyncGetBlob(tag, name, off, size, 0, out.data());
  f.Wait();
  if (f.get() == nullptr || f->GetReturnCode() != 0) return std::string();
  return out;
}

/** Blob size on one pool, or -1 when that pool does not have it. */
static long long SizeOn(const clio::run::PoolId &pool, const TagId &tag,
                        const std::string &name) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncGetBlobSize(tag, name, clio::run::PoolQuery::Local());
  f.Wait();
  return f->GetReturnCode() == 0 ? static_cast<long long>(f->size_) : -1;
}

static clio::run::u32 FlushCache(const clio::run::PoolId &pool) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncFlushData(clio::run::PoolQuery::Local(), 0);
  f.Wait();
  return f->GetReturnCode();
}

TEST_CASE("Config - durability: stated wins, else inferred from bdevs",
          "[cte][policy_cache][config]") {
  const std::string pers_file = R"(
  - path: "/tmp/dur.dat"
    bdev_type: "file"
    capacity_limit: "1MB"
    persistence_level: "long_term"
)";
  const std::string vol_ram = R"(
  - path: "ram::dur"
    bdev_type: "ram"
    capacity_limit: "1MB"
)";
  clio::cte::core::CoreDurability d;
  std::string err;
  auto resolve = [&](const std::string &yaml) {
    clio::cte::core::Config c;
    REQUIRE(c.LoadFromString(yaml));
    REQUIRE(c.ResolveDurability(&d, &err));
    return d;
  };
  using clio::cte::core::CoreDurability;
  // Unstated: persistent only when every bdev is persistent.
  REQUIRE(resolve("storage:" + pers_file) == CoreDurability::kPersistent);
  REQUIRE(resolve("storage:" + vol_ram) == CoreDurability::kVolatile);
  REQUIRE(resolve("storage:" + pers_file + vol_ram) ==
          CoreDurability::kVolatile);
  // A file bdev with no persistence_level is volatile.
  REQUIRE(resolve(R"(storage:
  - path: "/tmp/dur2.dat"
    bdev_type: "file"
    capacity_limit: "1MB"
)") == CoreDurability::kVolatile);
  REQUIRE(resolve("dpe:\n  dpe_type: random\n") == CoreDurability::kVolatile);
  // Stated wins over the bdevs (they are checked as they register).
  REQUIRE(resolve("durability: volatile\nstorage:" + pers_file) ==
          CoreDurability::kVolatile);
  REQUIRE(resolve("durability: persistent\nstorage:" + vol_ram) ==
          CoreDurability::kPersistent);
  // An unknown class is a config error.
  clio::cte::core::Config bad;
  REQUIRE_FALSE(bad.LoadFromString("durability: sometimes\nstorage:" + vol_ram));
}

TEST_CASE("Volatile core runs without a WAL", "[cte][policy_cache][wal]") {
  auto *fx = Fixture();
  // The volatile core was given metadata_log_path; it must create nothing
  // there, while the persistent core's log shards exist.
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_vol_log") == 0);
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_pers_log") > 0);
}

TEST_CASE("Write-back: put acks from volatile, flush makes it durable",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_basic");
  const std::string v = Payload(256 * 1024, 'a');
  REQUIRE(Put(kBackPool, tag, "b0", 0, v) == 0);
  REQUIRE(SizeOn(kVolatilePool, tag, "b0") == (long long)v.size());
  REQUIRE(SizeOn(kPersistentPool, tag, "b0") == -1);  // dirty, not flushed
  REQUIRE(Get(kBackPool, tag, "b0", 0, v.size()) == v);
  // A partial overwrite of the cached copy stays in volatile.
  const std::string patch = Payload(4096, 'q');
  REQUIRE(Put(kBackPool, tag, "b0", 8192, patch) == 0);
  std::string want = v;
  want.replace(8192, patch.size(), patch);
  REQUIRE(Get(kBackPool, tag, "b0", 0, v.size()) == want);

  REQUIRE(FlushCache(kBackPool) == 0);
  REQUIRE(SizeOn(kPersistentPool, tag, "b0") == (long long)v.size());
  REQUIRE(Get(kPersistentPool, tag, "b0", 0, v.size()) == want);
}

TEST_CASE("Write-back: a partial write to a new blob is absorbed",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_partial_new");
  // Not from offset 0: before, this went synchronously to persistent.
  const std::string data = Payload(4096, 'n');
  REQUIRE(Put(kBackPool, tag, "p0", 4096, data) == 0);
  REQUIRE(SizeOn(kVolatilePool, tag, "p0") == 8192);
  REQUIRE(SizeOn(kPersistentPool, tag, "p0") == -1);  // dirty, not flushed
  std::string want(4096, '\0');
  want += data;  // the gap before the write reads as zeros
  REQUIRE(Get(kBackPool, tag, "p0", 0, want.size()) == want);

  REQUIRE(FlushCache(kBackPool) == 0);
  REQUIRE(SizeOn(kPersistentPool, tag, "p0") == (long long)want.size());
  REQUIRE(Get(kPersistentPool, tag, "p0", 0, want.size()) == want);
}

TEST_CASE("Write-back: a partial overwrite of a persistent blob is absorbed",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_partial_old");
  // The blob exists only on persistent (written below the cache).
  const std::string base = Payload(64 * 1024, 'b');
  REQUIRE(Put(kPersistentPool, tag, "o0", 0, base) == 0);
  REQUIRE(SizeOn(kVolatilePool, tag, "o0") == -1);
  const std::string patch = Payload(500, 'x');
  REQUIRE(Put(kBackPool, tag, "o0", 1000, patch) == 0);
  // Fetch-on-write: the copy holds the whole blob with the patch applied,
  // and persistent is untouched until the flush.
  std::string want = base;
  want.replace(1000, patch.size(), patch);
  REQUIRE(SizeOn(kVolatilePool, tag, "o0") == (long long)base.size());
  REQUIRE(Get(kBackPool, tag, "o0", 0, base.size()) == want);
  REQUIRE(Get(kPersistentPool, tag, "o0", 0, base.size()) == base);

  REQUIRE(FlushCache(kBackPool) == 0);
  REQUIRE(Get(kPersistentPool, tag, "o0", 0, base.size()) == want);
}

TEST_CASE("Write-back: a partial write past a persistent blob's end",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_partial_grow");
  const std::string base = Payload(16 * 1024, 'g');
  REQUIRE(Put(kPersistentPool, tag, "g0", 0, base) == 0);
  const std::string tail = Payload(8192, 't');
  REQUIRE(Put(kBackPool, tag, "g0", 12 * 1024, tail) == 0);
  std::string want = base.substr(0, 12 * 1024) + tail;
  REQUIRE(SizeOn(kVolatilePool, tag, "g0") == (long long)want.size());
  REQUIRE(Get(kBackPool, tag, "g0", 0, want.size()) == want);

  REQUIRE(FlushCache(kBackPool) == 0);
  REQUIRE(SizeOn(kPersistentPool, tag, "g0") == (long long)want.size());
  REQUIRE(Get(kPersistentPool, tag, "g0", 0, want.size()) == want);
}

TEST_CASE("Write-back: concurrent partial writes under eviction are exact",
          "[cte][policy_cache][write_back][stress]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_partial_stress");
  // 48 x 1 MiB blobs >> the 8 MiB volatile core: every blob is created,
  // evicted (flush first), re-fetched and falls back to persistent while
  // other threads write other ranges of the same blobs.
  constexpr int kBlobs = 48;
  constexpr clio::run::u64 kBlob = kMiB;
  constexpr clio::run::u64 kChunk = 64 * 1024;
  constexpr int kChunks = static_cast<int>(kBlob / kChunk);
  constexpr int kThreads = 8;
  std::vector<std::pair<int, int>> work;  // (blob, chunk), shuffled
  for (int b = 0; b < kBlobs; ++b) {
    for (int c = 0; c < kChunks; ++c) work.emplace_back(b, c);
  }
  std::mt19937 rng(1234);
  std::shuffle(work.begin(), work.end(), rng);
  auto chunk_bytes = [](int b, int c) {
    return Payload(kChunk, static_cast<char>('A' + (b * 7 + c) % 50));
  };
  std::atomic<size_t> next{0};
  std::atomic<int> put_fail{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < work.size();) {
        const auto [b, c] = work[i];
        if (Put(kBackPool, tag, "s" + std::to_string(b), c * kChunk,
                chunk_bytes(b, c)) != 0) {
          put_fail.fetch_add(1);
        }
      }
    });
  }
  for (auto &th : threads) th.join();
  REQUIRE(put_fail.load() == 0);
  auto expect = [&](int b) {
    std::string want;
    for (int c = 0; c < kChunks; ++c) want += chunk_bytes(b, c);
    return want;
  };
  int bad_cache = 0;
  for (int b = 0; b < kBlobs; ++b) {
    if (Get(kBackPool, tag, "s" + std::to_string(b), 0, kBlob) != expect(b)) {
      ++bad_cache;
    }
  }
  INFO("blobs read back wrong through the cache: " << bad_cache);
  REQUIRE(bad_cache == 0);
  REQUIRE(FlushCache(kBackPool) == 0);
  int bad_pers = 0;
  for (int b = 0; b < kBlobs; ++b) {
    if (Get(kPersistentPool, tag, "s" + std::to_string(b), 0, kBlob) !=
        expect(b)) {
      ++bad_pers;
    }
  }
  INFO("blobs wrong on persistent after the flush: " << bad_pers);
  REQUIRE(bad_pers == 0);
}

TEST_CASE("Write-back: a batch of partial writes is absorbed",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_batch");
  // m0 exists on persistent; m1 is new. The batch patches m0 twice (in
  // order, overlapping) and writes m1 from a nonzero offset, as the FUSE
  // adapter's write-behind batches do.
  const std::string base = Payload(64 * 1024, 'm');
  REQUIRE(Put(kPersistentPool, tag, "m0", 0, base) == 0);
  const std::string a = Payload(4096, 'x');
  const std::string b = Payload(2048, 'y');
  const std::string c = Payload(4096, 'z');
  const std::string staged = a + b + c;
  std::vector<clio::cte::core::MultiPutDesc> descs(3);
  descs[0] = {tag, "m0", 8192, a.size(), 0};
  descs[1] = {tag, "m0", 9216, b.size(), a.size()};
  descs[2] = {tag, "m1", 4096, c.size(), a.size() + b.size()};
  auto buf = CLIO_IPC->AllocateBuffer(staged.size());
  REQUIRE(!buf.IsNull());
  memcpy(buf.ptr_, staged.data(), staged.size());
  clio::cte::core::Client back(kBackPool);
  auto f = back.AsyncMultiPutVectored(buf.shm_.template Cast<void>(),
                                      staged.size(), descs);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  REQUIRE(f->num_ok_ == 3);

  std::string want0 = base;
  want0.replace(8192, a.size(), a);
  want0.replace(9216, b.size(), b);
  std::string want1(4096, '\0');
  want1 += c;
  // Both land in the cache; persistent still has the old m0 and no m1.
  REQUIRE(SizeOn(kVolatilePool, tag, "m0") == (long long)base.size());
  REQUIRE(SizeOn(kPersistentPool, tag, "m1") == -1);
  REQUIRE(Get(kPersistentPool, tag, "m0", 0, base.size()) == base);
  REQUIRE(Get(kBackPool, tag, "m0", 0, base.size()) == want0);
  REQUIRE(Get(kBackPool, tag, "m1", 0, want1.size()) == want1);

  REQUIRE(FlushCache(kBackPool) == 0);
  REQUIRE(Get(kPersistentPool, tag, "m0", 0, base.size()) == want0);
  REQUIRE(Get(kPersistentPool, tag, "m1", 0, want1.size()) == want1);
}

TEST_CASE("Write-back: SyncTag flushes the tag's dirty copies",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_sync");
  const std::string v = Payload(64 * 1024, 'c');
  REQUIRE(Put(kBackPool, tag, "s0", 0, v) == 0);
  REQUIRE(SizeOn(kPersistentPool, tag, "s0") == -1);
  clio::cte::core::Client c(kBackPool);
  auto f = c.AsyncSyncTag(tag);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  REQUIRE(Get(kPersistentPool, tag, "s0", 0, v.size()) == v);
}

TEST_CASE("Write-back: deleting a never-flushed blob succeeds",
          "[cte][policy_cache][write_back]") {
  Fixture();
  const TagId tag = MakeTag("pc_wb_del");
  REQUIRE(Put(kBackPool, tag, "d0", 0, Payload(4096, 'd')) == 0);
  clio::cte::core::Client c(kBackPool);
  auto f = c.AsyncDelBlob(tag, "d0");
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  REQUIRE(SizeOn(kVolatilePool, tag, "d0") == -1);
  REQUIRE(SizeOn(kBackPool, tag, "d0") == -1);
}

TEST_CASE("Write-through: persistent first, copy kept current",
          "[cte][policy_cache][write_thru]") {
  Fixture();
  const TagId tag = MakeTag("pc_wt_basic");
  const std::string v1 = Payload(128 * 1024, 'e');
  REQUIRE(Put(kThruPool, tag, "t0", 0, v1) == 0);
  REQUIRE(Get(kPersistentPool, tag, "t0", 0, v1.size()) == v1);
  REQUIRE(SizeOn(kVolatilePool, tag, "t0") == (long long)v1.size());
  // A partial write reaches both; the copy never serves stale bytes.
  const std::string patch = Payload(1000, 'z');
  REQUIRE(Put(kThruPool, tag, "t0", 500, patch) == 0);
  std::string want = v1;
  want.replace(500, patch.size(), patch);
  REQUIRE(Get(kThruPool, tag, "t0", 0, v1.size()) == want);
  REQUIRE(Get(kVolatilePool, tag, "t0", 0, v1.size()) == want);
  REQUIRE(Get(kPersistentPool, tag, "t0", 0, v1.size()) == want);
}

TEST_CASE("Read miss fills the volatile core",
          "[cte][policy_cache][fill]") {
  Fixture();
  const TagId tag = MakeTag("pc_fill");
  const std::string v = Payload(300 * 1024, 'f');
  REQUIRE(Put(kPersistentPool, tag, "f0", 0, v) == 0);  // behind the cache
  REQUIRE(SizeOn(kVolatilePool, tag, "f0") == -1);
  REQUIRE(Get(kThruPool, tag, "f0", 0, v.size()) == v);   // miss -> fill
  REQUIRE(SizeOn(kVolatilePool, tag, "f0") == (long long)v.size());
  REQUIRE(Get(kThruPool, tag, "f0", 0, v.size()) == v);   // hit
  // A partial read fills too (chunked from persistent).
  REQUIRE(Put(kPersistentPool, tag, "f1", 0, v) == 0);
  REQUIRE(Get(kThruPool, tag, "f1", 1000, 2000) == v.substr(1000, 2000));
  REQUIRE(Get(kVolatilePool, tag, "f1", 0, v.size()) == v);
}

TEST_CASE("Write-back: eviction under a small volatile core loses nothing",
          "[cte][policy_cache][evict]") {
  Fixture();
  const TagId tag = MakeTag("pc_evict");
  // 12 x 1 MiB through an 8 MiB volatile core: early copies must be
  // flushed before they are evicted.
  std::vector<std::string> vals;
  for (int i = 0; i < 12; ++i) {
    vals.push_back(Payload(kMiB, static_cast<char>('A' + i)));
    REQUIRE(Put(kBackPool, tag, "e" + std::to_string(i), 0, vals.back()) == 0);
  }
  for (int i = 0; i < 12; ++i) {
    REQUIRE(Get(kBackPool, tag, "e" + std::to_string(i), 0, kMiB) == vals[i]);
  }
  REQUIRE(FlushCache(kBackPool) == 0);
  for (int i = 0; i < 12; ++i) {
    REQUIRE(Get(kPersistentPool, tag, "e" + std::to_string(i), 0, kMiB) ==
            vals[i]);
  }
}

TEST_CASE("Write-back: volatile-full fallback to persistent",
          "[cte][policy_cache][fallback]") {
  Fixture();
  const TagId tag = MakeTag("pc_fallback");
  // Larger than the whole volatile tier: no eviction can make it fit.
  const std::string big = Payload(kVolatileBytes + 4 * kMiB, 'k');
  REQUIRE(Put(kBackPool, tag, "big", 0, big) == 0);
  REQUIRE(SizeOn(kVolatilePool, tag, "big") == -1);
  REQUIRE(Get(kPersistentPool, tag, "big", 0, big.size()) == big);
  REQUIRE(Get(kBackPool, tag, "big", 0, big.size()) == big);

  // The same put with fallback off fails with the no-space code.
  const clio::run::u32 rc = Put(kStrictPool, tag, "big_strict", 0, big);
  REQUIRE(rc == clio::cte::core::kPutNoSpaceRc);
  REQUIRE(SizeOn(kPersistentPool, tag, "big_strict") == -1);
}

/** SetDurability on one core; returns rc and stores the previous class. */
static clio::run::u32 SetDur(const clio::run::PoolId &pool, clio::run::u32 want,
                             clio::run::u32 *previous = nullptr) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncSetDurability(want);
  f.Wait();
  if (previous != nullptr) *previous = f->previous_;
  return f->GetReturnCode();
}

/** Register a target on `pool` at runtime with an explicit level. */
static clio::run::u32 RegisterAt(const clio::run::PoolId &pool,
                                 const std::string &path,
                                 clio::run::bdev::BdevType type,
                                 const clio::run::PoolId &bdev_id,
                                 clio::run::i32 level) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncRegisterTarget(path, type, 4 * kMiB,
                                 clio::run::PoolQuery::Local(), bdev_id,
                                 clio::run::PoolQuery::Local(), 0,
                                 clio::run::u64(1) << 30, level);
  f.Wait();
  return f->GetReturnCode();
}

TEST_CASE("Durability - inferred at Create from the bdevs",
          "[cte][policy_cache][durability]") {
  auto *fx = Fixture();
  using namespace clio::cte::core;
  clio::run::u32 prev = 99;
  // All bdevs persistent: persistent (asking again changes nothing).
  REQUIRE(SetDur(kInferPersPool, kCteDurabilityPersistent, &prev) == 0);
  REQUIRE(prev == kCteDurabilityPersistent);
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_dur_pers_log") > 0);
  // A volatile bdev among them: volatile, no WAL written, and it cannot be
  // made persistent while that bdev is registered.
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_dur_mixed_log") == 0);
  REQUIRE(SetDur(kInferVolPool, kCteDurabilityPersistent, &prev) ==
          kCteVolatileTargetRc);
  REQUIRE(prev == kCteDurabilityVolatile);
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_dur_mixed_log") == 0);
}

TEST_CASE("Durability - RegisterTarget checks the bdev against the core",
          "[cte][policy_cache][durability]") {
  auto *fx = Fixture();
  using namespace clio::cte::core;
  using clio::run::bdev::BdevType;
  // A persistent core refuses a volatile bdev, whether the level is given
  // or defaulted (an unlisted target is volatile).
  REQUIRE(RegisterAt(kInferPersPool, "ram::policy_cache_dur_reject", BdevType::kRam,
                     clio::run::PoolId(580, 50), -1) == kCteVolatileTargetRc);
  REQUIRE(RegisterAt(kInferPersPool, "ram::policy_cache_dur_reject2",
                     BdevType::kRam, clio::run::PoolId(580, 51),
                     0) == kCteVolatileTargetRc);
  // A volatile core accepts a persistent bdev (with a warning).
  REQUIRE(RegisterAt(kSwitchPool, fx->p_.dur_extra_file, BdevType::kFile,
                     clio::run::PoolId(590, 50), 1) == 0);
}

TEST_CASE("Durability - SetDurability switches the WAL at runtime",
          "[cte][policy_cache][durability]") {
  auto *fx = Fixture();
  using namespace clio::cte::core;
  clio::cte::core::Client sw(kSwitchPool);
  auto tf = sw.AsyncGetOrCreateTag("dur_switch_tag");
  tf.Wait();
  REQUIRE(tf->GetReturnCode() == 0);
  const TagId tag = tf->tag_id_;
  const std::string a = Payload(64 * 1024, 'a');
  // Volatile (stated): puts work and no WAL is written.
  REQUIRE(Put(kSwitchPool, tag, "while_volatile", 0, a) == 0);
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_dur_switch_log") == 0);
  // To persistent: the WAL opens and a snapshot is written.
  clio::run::u32 prev = 99;
  REQUIRE(SetDur(kSwitchPool, kCteDurabilityPersistent, &prev) == 0);
  REQUIRE(prev == kCteDurabilityVolatile);
  REQUIRE(CountPrefixed(fx->p_.dir, "policy_cache_dur_switch_log") > 0);
  REQUIRE(Put(kSwitchPool, tag, "while_persistent", 0, a) == 0);
  REQUIRE(Get(kSwitchPool, tag, "while_volatile", 0, a.size()) == a);
  // A persistent core refuses an unknown class and volatile bdevs.
  REQUIRE(SetDur(kSwitchPool, 7) == kCteInvalidDurabilityRc);
  REQUIRE(RegisterAt(kSwitchPool, "ram::policy_cache_dur_sw_ram",
                     clio::run::bdev::BdevType::kRam, clio::run::PoolId(590, 51),
                     -1) == kCteVolatileTargetRc);
  // Back to volatile; then a volatile bdev registers, after which the core
  // cannot become persistent again.
  REQUIRE(SetDur(kSwitchPool, kCteDurabilityVolatile, &prev) == 0);
  REQUIRE(prev == kCteDurabilityPersistent);
  REQUIRE(RegisterAt(kSwitchPool, "ram::policy_cache_dur_sw_ram",
                     clio::run::bdev::BdevType::kRam, clio::run::PoolId(590, 52),
                     -1) == 0);
  REQUIRE(SetDur(kSwitchPool, kCteDurabilityPersistent) == kCteVolatileTargetRc);
  REQUIRE(Get(kSwitchPool, tag, "while_persistent", 0, a.size()) == a);
}

SIMPLE_TEST_MAIN()
