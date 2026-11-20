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
 * Unplugging ("stopping") a write-back policy cache.
 *
 * One in-process runtime composes two DISCONNECTED cores -- persistent 512.0
 * (file tier, WAL on) and volatile 540.0 (RAM tier) -- and every test case
 * creates its OWN write-back cache pool over them (distinct id and name),
 * because an unplug destroys the pool. The caches never flush on their own
 * (no period, no fill trigger, no continuous trigger), so what is durable
 * after a case is exactly what the unplug made durable. The persistent core
 * is read directly to tell where the bytes are.
 *
 * Decisions taken from reading the admin DestroyPool code:
 *  - Admin counters move in phase two only for `unplugs` and
 *    `redirects_installed`; the flushed counters move in phase one.
 *  - A pool that does not exist makes DestroyPool log an error but return
 *    rc 0, so that case only asserts "no crash, runtime still answers".
 */

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_config.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core_persist_write_back/core_persist_write_back_client.h>
#include "clio_ctp/serialize/msgpack_wrapper.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

/**
 * Convert a msgpack object to T (the CTP msgpack shim has convert(), not as<>).
 * @param o the object
 * @return its value as T
 */
template <typename T>
T MsgAs(const msgpack::object &o) {
  T out{};
  o.convert(out);
  return out;
}


namespace fs = std::filesystem;
using clio::cte::core::TagId;
using clio::run::PoolId;
using clio::run::u32;
using clio::run::u64;

namespace {

/** Name prefix of every file this test creates under the test dir. */
const char kPrefix[] = "cache_unplug_";
const PoolId kPersistentPool(512, 0);
const PoolId kVolatilePool(540, 0);
constexpr u64 kKiB = 1024;
constexpr u64 kMiB = 1024 * kKiB;
/** Level flags, shorthand for the admin constants. */
const u32 kStop = clio::run::admin::kDestroyPoolKeepRestartable;
const u32 kForce = clio::run::admin::kDestroyPoolNoFlush;
const u32 kRm = clio::run::admin::kDestroyPoolDeleteData;

/** Decoded numeric fields of a Monitor "stats" map (non-numeric skipped). */
using StatMap = std::map<std::string, u64>;

/** @return the directory tests may write to */
std::string TestDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Remove every entry under `dir` whose name starts with `prefix`. */
void RemovePrefixed(const std::string &dir, const std::string &prefix) {
  std::error_code ec;
  for (const auto &e : fs::directory_iterator(dir, ec)) {
    if (e.path().filename().string().rfind(prefix, 0) == 0) {
      fs::remove_all(e.path(), ec);
    }
  }
}

/**
 * Write the compose config: a persistent core and a volatile core.
 * @param dir directory for the config, file tier and metadata log
 * @return the config path
 */
std::string WriteConfig(const std::string &dir) {
  const std::string config = dir + "/" + kPrefix + "config.yaml";
  std::ofstream f(config);
  REQUIRE(f.is_open());
  f << R"(
runtime:
  num_threads: 2
  queue_depth: 1024

compose:
  - mod_name: clio_cte_core
    pool_name: cache_unplug_persistent
    pool_query: local
    pool_id: 512.0
    durability: persistent
    targets:
      neighborhood: 1
    storage:
      - path: ")" << dir << "/" << kPrefix << R"(file.dat"
        bdev_type: "file"
        capacity_limit: "512MB"
        score: 0.2
        persistence_level: "temporary"
    performance:
      metadata_log_path: ")" << dir << "/" << kPrefix << R"(pers_log"

  - mod_name: clio_cte_core
    pool_name: cache_unplug_volatile
    pool_query: local
    pool_id: 540.0
    targets:
      neighborhood: 1
    storage:
      - path: "ram::cache_unplug_vram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0
)";
  return config;
}

/** Runtime + cores, started once for the whole binary. */
class UnplugFixture {
 public:
  UnplugFixture() {
    const std::string dir = TestDir();
    RemovePrefixed(dir, kPrefix);
    const std::string config = WriteConfig(dir);
    const std::string restart = dir + "/" + kPrefix + "restart.bin";
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  }
};

/** @return the fixture, creating it (and the runtime) on first use */
UnplugFixture *Fixture() {
  static UnplugFixture *f = new UnplugFixture();
  return f;
}

/**
 * Create a write-back cache that flushes only when asked.
 * @param id pool id of the cache
 * @param name pool name of the cache
 */
void CreateCache(const PoolId &id, const std::string &name) {
  clio::cte::core_persist_write_back::Client c;
  clio::cte::core_persist_write_back::WriteBackCacheConfig params;
  params.volatile_pool_ = kVolatilePool;
  params.persistent_pool_ = kPersistentPool;
  params.flush_period_ms_ = 0;
  params.flush_fill_pct_ = 0;
  params.flush_dirty_mb_ = 0;
  auto f = c.AsyncCreateCache(clio::run::PoolQuery::Local(), name, id, params);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
}

/** A tag created on the persistent core (the tag authority). */
TagId MakeTag(const std::string &name) {
  clio::cte::core::Client pers(kPersistentPool);
  auto f = pers.AsyncGetOrCreateTag(name);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  return f->tag_id_;
}

/** Deterministic payload; `seed` changes every byte. */
std::string Payload(u64 size, char seed) {
  std::string v(size, '\0');
  for (u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>(seed + (i * 7) % 61);
  }
  return v;
}

/** Put [off, off+data.size()) through `pool`; the task's return code. */
u32 Put(const PoolId &pool, const TagId &tag, const std::string &name, u64 off,
        const std::string &data) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncPutBlob(tag, name, off, data.size(), data.data());
  f.Wait();
  return f.get() == nullptr ? 1u : f->GetReturnCode();
}

/** Read [off, off+size) through `pool`; empty string when the read fails. */
std::string Get(const PoolId &pool, const TagId &tag, const std::string &name,
                u64 off, u64 size) {
  clio::cte::core::Client c(pool);
  std::string out(size, '\0');
  auto f = c.AsyncGetBlob(tag, name, off, size, 0, out.data());
  f.Wait();
  if (f.get() == nullptr || f->GetReturnCode() != 0) return std::string();
  return out;
}

/** Blob size on one pool, or -1 when that pool does not have it. */
long long SizeOn(const PoolId &pool, const TagId &tag,
                 const std::string &name) {
  clio::cte::core::Client c(pool);
  auto f = c.AsyncGetBlobSize(tag, name, clio::run::PoolQuery::Local());
  f.Wait();
  return f->GetReturnCode() == 0 ? static_cast<long long>(f->size_) : -1;
}

/**
 * Decode the msgpack map(s) of a Monitor "stats" result; numeric values of
 * every container are summed per key.
 * @param f a completed MonitorTask future
 * @return the numeric fields
 */
template <typename FutureT>
StatMap DecodeStats(FutureT &f) {
  StatMap out;
  REQUIRE(f->GetReturnCode() == 0);
  for (const auto &kv : f->results_) {
    const auto &blob = kv.second;
    if (blob.empty()) continue;
    msgpack::object_handle oh = msgpack::unpack(blob.data(), blob.size());
    const msgpack::object &o = oh.get();
    REQUIRE(o.type == msgpack::type::MAP);
    for (uint32_t i = 0; i < o.via.map.size; ++i) {
      const auto &entry = o.via.map.ptr[i];
      if (entry.val.type == msgpack::type::POSITIVE_INTEGER) {
        out[MsgAs<std::string>(entry.key)] += entry.val.via.u64;
      }
    }
  }
  return out;
}

/** Monitor "stats" of the admin pool. */
StatMap AdminStats() {
  auto *admin = CLIO_ADMIN;
  REQUIRE(admin != nullptr);
  auto f = admin->AsyncMonitor(clio::run::PoolQuery::Local(), "stats");
  f.Wait();
  return DecodeStats(f);
}

/** Monitor "stats" of a cache pool. */
StatMap CacheStats(const PoolId &cache) {
  clio::cte::core::Client c(cache);
  auto f = c.AsyncMonitor(clio::run::PoolQuery::Local(), "stats");
  f.Wait();
  return DecodeStats(f);
}

/** Unplug result: rc plus the counts the admin reported. */
struct UnplugResult {
  u32 rc = 0;
  u64 blobs = 0;
  u64 bytes = 0;
  std::string err;
};

/** Run an UnplugPool at `flags` against `pool`. */
UnplugResult Unplug(const PoolId &pool, u32 flags) {
  auto *admin = CLIO_ADMIN;
  REQUIRE(admin != nullptr);
  UnplugResult r;
  r.rc = admin->UnplugPool(pool, flags, &r.err, &r.blobs, &r.bytes);
  return r;
}

/** One blob a case writes: name and full expected content. */
struct Blob {
  std::string name;
  std::string data;
};

/** Sizes cycled through by MakeBlobs: 4 KiB up to 1 MiB, some unaligned. */
const u64 kSizes[] = {4 * kKiB, 12345, 64 * kKiB, 200000, 512 * kKiB, kMiB};

/**
 * Build `count` blobs with distinct names, patterns and mixed sizes.
 * @param prefix blob name prefix
 * @param count number of blobs
 */
std::vector<Blob> MakeBlobs(const std::string &prefix, int count) {
  std::vector<Blob> v;
  for (int i = 0; i < count; ++i) {
    const u64 size = kSizes[i % (sizeof(kSizes) / sizeof(kSizes[0]))];
    v.push_back({prefix + std::to_string(i),
                 Payload(size, static_cast<char>('A' + i % 40))});
  }
  return v;
}

/** Put every blob through `pool`, requiring rc 0. */
void PutAll(const PoolId &pool, const TagId &tag,
            const std::vector<Blob> &blobs) {
  for (const auto &b : blobs) {
    REQUIRE(Put(pool, tag, b.name, 0, b.data) == 0);
  }
}

/** Sum of the blobs' sizes. */
u64 TotalBytes(const std::vector<Blob> &blobs) {
  u64 n = 0;
  for (const auto &b : blobs) n += b.data.size();
  return n;
}

/** Require every blob to read back exactly from `pool`. */
void RequireAllOn(const PoolId &pool, const TagId &tag,
                  const std::vector<Blob> &blobs) {
  for (const auto &b : blobs) {
    INFO("blob " << b.name);
    REQUIRE(SizeOn(pool, tag, b.name) == (long long)b.data.size());
    REQUIRE(Get(pool, tag, b.name, 0, b.data.size()) == b.data);
  }
}

/** Bytes of the patch WriteDirtySet writes into the persistent-held blob. */
constexpr u64 kExistingPatchBytes = 1000;

/**
 * Write the standard dirty set through `cache`: blobs "<prefix>0..count-2"
 * are new, blob 0 then gets a partial overwrite, and the last blob already
 * exists on the persistent core (old content) and is patched through the
 * cache. Distinct dirty blobs: `count`.
 * @param blobs in/out: expected FINAL content of every blob
 * @param old_last out: the persistent core's content of the last blob before
 */
void WriteDirtySet(const PoolId &cache, const TagId &tag,
                   const std::string &prefix, int count,
                   std::vector<Blob> *blobs, std::string *old_last) {
  *blobs = MakeBlobs(prefix, count);
  Blob &last = blobs->back();
  *old_last = Payload(last.data.size(), 'z');
  REQUIRE(Put(kPersistentPool, tag, last.name, 0, *old_last) == 0);
  // New blobs, then the existing one patched through the cache.
  for (size_t i = 0; i + 1 < blobs->size(); ++i) {
    REQUIRE(Put(cache, tag, (*blobs)[i].name, 0, (*blobs)[i].data) == 0);
  }
  const std::string patch = Payload(kExistingPatchBytes, 'k');
  std::string want_last = *old_last;
  want_last.replace(2000, patch.size(), patch);
  REQUIRE(Put(cache, tag, last.name, 2000, patch) == 0);
  last.data = want_last;
  // A partial overwrite of a blob already buffered in the cache.
  Blob &first = blobs->front();
  const std::string patch0 = Payload(500, 'p');
  REQUIRE(Put(cache, tag, first.name, 100, patch0) == 0);
  first.data.replace(100, patch0.size(), patch0);
}

}  // namespace

TEST_CASE("Unplug stop flushes every dirty blob to the persistent core",
          "[cte][cache_unplug][stop]") {
  Fixture();
  const PoolId cache(600, 0);
  CreateCache(cache, "unplug_stop");
  const TagId tag = MakeTag("cu_stop");
  constexpr int kBlobs = 64;
  std::vector<Blob> blobs;
  std::string old_last;
  WriteDirtySet(cache, tag, "s", kBlobs, &blobs, &old_last);

  // Dirty in the cache, not yet durable.
  const StatMap before_cache = CacheStats(cache);
  REQUIRE(before_cache.at("dirty_blobs") == (u64)kBlobs);
  for (size_t i = 0; i + 1 < blobs.size(); ++i) {
    INFO("blob " << blobs[i].name);
    REQUIRE(SizeOn(kPersistentPool, tag, blobs[i].name) == -1);
  }
  REQUIRE(Get(kPersistentPool, tag, blobs.back().name, 0, old_last.size()) ==
          old_last);
  // Dirty bytes are the written ranges: every new blob whole, plus only the
  // 1000-byte patch of the blob persistent already held (its other bytes
  // were filled from persistent, so they are clean).
  const u64 want_bytes = before_cache.at("dirty_bytes");
  const u64 written =
      TotalBytes(blobs) - blobs.back().data.size() + kExistingPatchBytes;
  INFO("dirty_bytes=" << want_bytes << " written bytes=" << written);
  REQUIRE(want_bytes == written);

  const StatMap admin_before = AdminStats();
  UnplugResult r = Unplug(cache, kStop);
  INFO("unplug err: " << r.err);
  REQUIRE(r.rc == 0);
  REQUIRE(r.blobs == (u64)kBlobs);
  REQUIRE(r.bytes == want_bytes);

  RequireAllOn(kPersistentPool, tag, blobs);
  const StatMap admin_after = AdminStats();
  REQUIRE(admin_after.at("unplugs") == admin_before.at("unplugs") + 1);
  REQUIRE(admin_after.at("unplug_blobs_flushed") ==
          admin_before.at("unplug_blobs_flushed") + r.blobs);
  REQUIRE(admin_after.at("unplug_bytes_flushed") ==
          admin_before.at("unplug_bytes_flushed") + r.bytes);
  REQUIRE(admin_after.at("unplug_flush_failures") ==
          admin_before.at("unplug_flush_failures"));
}

TEST_CASE("Unplug leaves a redirect: tasks to the cache pool reach persistent",
          "[cte][cache_unplug][redirect]") {
  Fixture();
  const PoolId cache(601, 0);
  CreateCache(cache, "unplug_redirect");
  const TagId tag = MakeTag("cu_redirect");
  std::vector<Blob> blobs = MakeBlobs("r", 4);
  PutAll(cache, tag, blobs);

  const StatMap admin_before = AdminStats();
  REQUIRE(Unplug(cache, kStop).rc == 0);
  REQUIRE(AdminStats().at("redirects_installed") >
          admin_before.at("redirects_installed"));

  // A client still pointed at the unplugged cache pool id.
  const std::string fresh = Payload(100 * kKiB, 'f');
  REQUIRE(Put(cache, tag, "after", 0, fresh) == 0);
  REQUIRE(SizeOn(kPersistentPool, tag, "after") == (long long)fresh.size());
  REQUIRE(Get(kPersistentPool, tag, "after", 0, fresh.size()) == fresh);
  REQUIRE(Get(cache, tag, "after", 0, fresh.size()) == fresh);
  // And the pre-unplug data is reachable through the old id too.
  REQUIRE(Get(cache, tag, blobs[0].name, 0, blobs[0].data.size()) ==
          blobs[0].data);
}

TEST_CASE("Unplug --force discards unflushed data", "[cte][cache_unplug][force]") {
  Fixture();
  const PoolId cache(602, 0);
  CreateCache(cache, "unplug_force");
  const TagId tag = MakeTag("cu_force");
  std::vector<Blob> blobs;
  std::string old_last;
  WriteDirtySet(cache, tag, "x", 16, &blobs, &old_last);
  REQUIRE(CacheStats(cache).at("dirty_blobs") == 16);

  UnplugResult r = Unplug(cache, kStop | kForce);
  INFO("unplug err: " << r.err);
  REQUIRE(r.rc == 0);
  REQUIRE(r.blobs == 0);
  for (size_t i = 0; i + 1 < blobs.size(); ++i) {
    INFO("blob " << blobs[i].name);
    REQUIRE(SizeOn(kPersistentPool, tag, blobs[i].name) == -1);
  }
  // The pre-existing blob keeps its OLD content, not the patched copy.
  REQUIRE(Get(kPersistentPool, tag, blobs.back().name, 0, old_last.size()) ==
          old_last);
}

TEST_CASE("Unplug rm flushes, then drops the volatile copies",
          "[cte][cache_unplug][rm]") {
  Fixture();
  const PoolId cache(603, 0);
  CreateCache(cache, "unplug_rm");
  const TagId tag = MakeTag("cu_rm");
  std::vector<Blob> blobs;
  std::string old_last;
  WriteDirtySet(cache, tag, "m", 16, &blobs, &old_last);
  // Before the rm the volatile core holds the buffered copies.
  for (const auto &b : blobs) {
    REQUIRE(SizeOn(kVolatilePool, tag, b.name) == (long long)b.data.size());
  }

  UnplugResult r = Unplug(cache, kRm);
  INFO("unplug err: " << r.err);
  REQUIRE(r.rc == 0);
  REQUIRE(r.blobs == 16);
  RequireAllOn(kPersistentPool, tag, blobs);
  for (const auto &b : blobs) {
    INFO("volatile copy of " << b.name);
    REQUIRE(SizeOn(kVolatilePool, tag, b.name) == -1);
    REQUIRE(Get(kVolatilePool, tag, b.name, 0, b.data.size()).empty());
  }
}

/** What one writer thread attempted for one blob name. */
struct Attempt {
  char seed;
  bool acked;
};

/** Per-thread record: attempts per blob name, in order. */
struct WriterLog {
  std::map<std::string, std::vector<Attempt>> attempts;
  std::map<std::string, u64> sizes;
};

TEST_CASE("Unplug racing concurrent writers loses no acked write",
          "[cte][cache_unplug][race]") {
  Fixture();
  const PoolId cache(604, 0);
  CreateCache(cache, "unplug_race");
  const TagId tag = MakeTag("cu_race");
  constexpr int kThreads = 4;
  constexpr int kNames = 8;
  constexpr int kMaxPuts = 1500;
  std::vector<WriterLog> logs(kThreads);
  std::atomic<bool> stop{false};
  std::atomic<u64> acked{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      WriterLog &log = logs[t];
      for (int i = 0; i < kMaxPuts && !stop.load(); ++i) {
        const std::string name =
            "w" + std::to_string(t) + "_" + std::to_string(i % kNames);
        const u64 size = 8 * kKiB * (1 + (i % kNames) % 4);
        const char seed = static_cast<char>('A' + i % 50);
        log.sizes[name] = size;
        const u32 rc = Put(cache, tag, name, 0, Payload(size, seed));
        log.attempts[name].push_back({seed, rc == 0});
        if (rc == 0) acked.fetch_add(1);
      }
    });
  }
  // Wait until the writers are mid-stream, then unplug under them.
  for (int i = 0; i < 600 && acked.load() < 40; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(acked.load() >= 40);
  UnplugResult r = Unplug(cache, kStop);
  INFO("unplug err: " << r.err);
  const u64 acked_at_unplug = acked.load();
  // Keep writing through the redirect for a while.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  stop.store(true);
  for (auto &th : threads) th.join();
  REQUIRE(r.rc == 0);
  INFO("acked at unplug=" << acked_at_unplug << " total=" << acked.load());

  // Every blob's persistent content is its last acked write, or a later
  // unacked attempt that may have landed anyway.
  for (int t = 0; t < kThreads; ++t) {
    for (const auto &kv : logs[t].attempts) {
      const std::string &name = kv.first;
      const std::vector<Attempt> &at = kv.second;
      const u64 size = logs[t].sizes.at(name);
      int last_acked = -1;
      for (int i = 0; i < (int)at.size(); ++i) {
        if (at[i].acked) last_acked = i;
      }
      if (last_acked < 0) continue;  // never acked: no promise
      const std::string got = Get(kPersistentPool, tag, name, 0, size);
      INFO("blob " << name << " attempts=" << at.size()
                   << " last_acked=" << last_acked);
      REQUIRE(!got.empty());
      bool ok = false;
      for (int i = last_acked; i < (int)at.size() && !ok; ++i) {
        ok = got == Payload(size, at[i].seed);
      }
      REQUIRE(ok);
    }
  }
}

TEST_CASE("Re-creating a cache with the same id clears the redirect",
          "[cte][cache_unplug][recreate]") {
  Fixture();
  const PoolId cache(605, 0);
  CreateCache(cache, "unplug_recreate");
  const TagId tag = MakeTag("cu_recreate");
  REQUIRE(Put(cache, tag, "before", 0, Payload(8 * kKiB, 'b')) == 0);
  REQUIRE(Unplug(cache, kStop).rc == 0);
  // Redirected while unplugged: goes straight to persistent.
  REQUIRE(Put(cache, tag, "mid", 0, Payload(8 * kKiB, 'm')) == 0);
  REQUIRE(SizeOn(kPersistentPool, tag, "mid") == (long long)(8 * kKiB));

  CreateCache(cache, "unplug_recreate");
  const std::string data = Payload(16 * kKiB, 'n');
  REQUIRE(Put(cache, tag, "after", 0, data) == 0);
  // Buffered again: dirty in the cache, absent from persistent.
  REQUIRE(CacheStats(cache).at("dirty_blobs") >= 1);
  REQUIRE(SizeOn(kPersistentPool, tag, "after") == -1);
  REQUIRE(Get(cache, tag, "after", 0, data.size()) == data);
}

TEST_CASE("Unplugging a missing pool is safe; unknown names resolve to null",
          "[cte][cache_unplug][missing]") {
  Fixture();
  auto *admin = CLIO_ADMIN;
  REQUIRE(admin != nullptr);
  REQUIRE(admin->ResolvePoolId("no_such_pool").IsNull());

  const PoolId cache(606, 0);
  CreateCache(cache, "unplug_twice");
  REQUIRE(admin->ResolvePoolId("unplug_twice") == cache);
  REQUIRE(admin->ResolvePoolId(cache.ToString()) == cache);
  const TagId tag = MakeTag("cu_twice");
  REQUIRE(Put(cache, tag, "t0", 0, Payload(4 * kKiB, 't')) == 0);
  UnplugResult first = Unplug(cache, kStop);
  REQUIRE(first.rc == 0);
  REQUIRE(first.blobs == 1);

  // Second stop of the gone pool, and a pool that never existed: DestroyPool
  // logs an error and returns 0, so rc is not asserted -- only that nothing
  // is flushed, nothing hangs or crashes, and the runtime still answers.
  UnplugResult again = Unplug(cache, kStop);
  REQUIRE(again.blobs == 0);
  UnplugResult never = Unplug(PoolId(999, 7), kStop);
  REQUIRE(never.blobs == 0);
  REQUIRE(AdminStats().count("unplugs") == 1);
  // The data flushed by the first stop is still intact.
  REQUIRE(SizeOn(kPersistentPool, tag, "t0") == (long long)(4 * kKiB));
}

SIMPLE_TEST_MAIN()
