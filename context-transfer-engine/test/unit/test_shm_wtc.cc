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
 * core client bound to the cache pool and connected to its segment (the shm
 * cache path of PutBlob/GetBlob/GetBlobSize); `core` is bound to the core
 * pool (task path) and reads the chain directly to check what landed.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>  // _getpid
#define getpid _getpid
#else
#include <unistd.h>  // getpid
#endif

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
    CleanupDeviceFilesOnce();
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

  /**
   * Remove the full-device core's backing file, allocator log and marker
   * left by an earlier run. Once per process, before the runtime opens the
   * device: every test case builds a fixture, and the device stays open.
   */
  static void CleanupDeviceFilesOnce() {
    static bool done = false;
    if (done) return;
    done = true;
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(DataDir(), ec)) {
      if (e.path().filename().string().rfind(kFullDeviceFile, 0) == 0) {
        fs::remove(e.path(), ec);
      }
    }
  }

  /** Backing file name (in DataDir) of the full-device core's target. */
  static constexpr const char *kFullDeviceFile = "shm_wtc_full.dat";

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

  # A core over one file device, for the out-of-space case.
  - mod_name: clio_cte_core
    pool_name: shm_wtc_full_core
    pool_query: local
    pool_id: 513.0
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
    storage:
      - path: ")" << DataDir() << "/" << kFullDeviceFile << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
  }
};

/**
 * How long a put refused for room waits for earlier ranges to flush in every
 * test cache (WtcConfig::nospace_wait_ms_; production default 250 ms). A
 * test asserts what lands in the cache, so whether a refused put is cached
 * or written around must not depend on how fast this machine flushes: a
 * slow runner's flusher can take longer than 250 ms to land the ranges ahead
 * of a put. A flush that never lands still fails (the put writes around
 * after this wait; the ctest TIMEOUT bounds the whole run).
 */
static constexpr clio::run::u32 kTestNoSpaceWaitMs = 60000;

/** Create a cache pool over the core (data_capacity > 0: extent mode, its
 *  bytes in granules leased from the core's RAM bdev on this node). */
static void CreateCache(const clio::run::PoolId &id, const std::string &name,
                        clio::run::u64 capacity, bool md_only = false,
                        clio::run::u64 data_capacity = 0,
                        double flush_period_us = 0,
                        const clio::run::PoolId &next =
                            clio::cte::core::kCtePoolId,
                        clio::run::u64 flush_batch_bytes = 0) {
  clio::cte::shm_wtc::Client admin;
  clio::cte::shm_wtc::WtcConfig params;
  params.md_only_ = md_only;
  if (data_capacity > 0) {
    params.data_pool_ = clio::run::PoolId(512, 0);  // this node's RAM tier
    params.data_capacity_ = data_capacity;
  }
  params.next_pool_id_ = next;
  params.capacity_ = capacity;
  params.max_blobs_ = 4096;
  params.log_depth_ = 4096;
  params.nospace_wait_ms_ = kTestNoSpaceWaitMs;
  if (flush_period_us > 0) params.flush_period_us_ = flush_period_us;
  if (flush_batch_bytes > 0) params.flush_batch_bytes_ = flush_batch_bytes;
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

/**
 * Hex dump of up to `n` bytes of `s` starting at `off`, for failure messages.
 * @param s   the buffer
 * @param off first byte
 * @param n   most bytes shown
 * @return space-separated hex bytes
 */
static std::string HexAt(const std::string &s, size_t off, size_t n) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  for (size_t i = off; i < s.size() && i < off + n; ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (!out.empty()) out += ' ';
    out += kHex[c >> 4];
    out += kHex[c & 0xF];
  }
  return out;
}

/**
 * Describe how a read differs from what was written, so one CI failure says
 * where the wrong bytes are and whose they look like.
 * @param want the bytes written (a Payload, for the seed guess)
 * @param got  the bytes read back
 * @return "" when equal; else sizes, the first and last differing offsets
 *         (and their 4 KiB granule), the differing byte count, both sides'
 *         bytes at the first difference, and the Payload seed the read
 *         bytes there fit (another blob's seed = its bytes; 0 bytes = a hole)
 */
static std::string DescribeMismatch(const std::string &want,
                                    const std::string &got) {
  if (want == got) return "";
  std::ostringstream os;
  os << "size want=" << want.size() << " got=" << got.size();
  const size_t n = std::min(want.size(), got.size());
  size_t first = n;
  size_t last = 0;
  size_t diff = 0;
  for (size_t i = 0; i < n; ++i) {
    if (want[i] == got[i]) continue;
    if (first == n) first = i;
    last = i;
    ++diff;
  }
  if (first == n) return os.str();
  const int seed = static_cast<unsigned char>(got[first]) -
                   static_cast<int>(first % 53);
  os << "; first diff at " << first << " (granule " << first / 4096
     << "), last at " << last << " (granule " << last / 4096 << "), "
     << diff << " bytes differ; want[" << first << "..]: "
     << HexAt(want, first, 16) << "; got[" << first << "..]: "
     << HexAt(got, first, 16) << "; got fits Payload seed "
     << ((seed % 256) + 256) % 256;
  return os.str();
}

/**
 * The keys of one cache pool, read from its Locator in the metadata
 * directory. TEST-ONLY: a test composes several caches over the same core,
 * and the core's Create names one of them; production clients take the keys
 * from the CTE Create (see the "keys come from the CTE Create" case).
 * @param pool the cache pool
 * @return the keys (invalid when the pool published no Locator)
 */
static clio::cte::core::ShmCacheKeys KeysOf(const clio::run::PoolId &pool) {
  clio::cte::core::ShmCacheKeys keys;
  auto *ipc = CLIO_IPC;
  auto *meta = ipc->GetMetadataAllocator();
  auto *dir = ipc->GetMetadataDirectory();
  if (meta == nullptr || dir == nullptr) return keys;
  const clio::run::u64 off = dir->FindRoot(pool.ToU64());
  if (off == 0) return keys;
  const auto *loc = reinterpret_cast<const wtc::Locator *>(
      reinterpret_cast<const char *>(meta) + off);
  if (loc->magic_ != wtc::Locator::kMagic || loc->ready_.load() != 1) {
    return keys;
  }
  keys.runtime_pid_ = loc->runtime_pid_;
  keys.cache_pool_ = pool.ToU64();
  keys.root_off_ = loc->root_off_;
  keys.entry_pool_ = loc->entry_pool_;
  return keys;
}

/**
 * The test's own mapping of a cache segment, for the flush barrier and for
 * invalidations a client never issues. One per pool, for the process.
 * @param keys keys of the cache
 * @return the view, or nullptr when the segment cannot be mapped
 */
static wtc::View *TestView(const clio::cte::core::ShmCacheKeys &keys) {
  static std::map<clio::run::u64, std::unique_ptr<wtc::View>> views;
  auto it = views.find(keys.cache_pool_);
  if (it != views.end()) return it->second.get();
  auto v = std::make_unique<wtc::View>();
  if (!v->Attach(wtc::SegmentName(keys.runtime_pid_,
                                  clio::run::PoolId::FromU64(keys.cache_pool_)),
                 keys.root_off_)) {
    return nullptr;
  }
  wtc::View *raw = v.get();
  views[keys.cache_pool_] = std::move(v);
  return raw;
}

/**
 * Connect a client bound to a cache pool to that cache's segment.
 * @param io client bound to the cache pool
 * @param pool the cache pool
 */
static void Connect(clio::cte::core::Client &io,
                    const clio::run::PoolId &pool) {
  REQUIRE(io.ConnectCache(KeysOf(pool)));
  REQUIRE(io.HasCache());
}

/** Longest WaitFlushed waits. Generous: tests call it after tens of MB of
 *  puts, which a slow runner's flusher may take many seconds to land; a
 *  flusher that never lands them still fails, only later. */
static constexpr double kFlushDeadlineMs = 120000.0;

/**
 * Wait (bounded by kFlushDeadlineMs) until the cache has flushed every range
 * pushed so far. Prints both counters when the deadline passes.
 * @param io client connected to the cache
 * @return true once flushed_ reaches pushed_; false on the deadline
 */
static bool WaitFlushed(clio::cte::core::Client &io) {
  wtc::View *view = TestView(io.CacheKeys());
  if (view == nullptr) return false;
  const wtc::Root *root = view->GetRoot();
  const auto t0 = std::chrono::steady_clock::now();
  while (root->flushed_.load() < root->pushed_.load()) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    if (ms >= kFlushDeadlineMs) {
      std::cout << "  WaitFlushed: gave up after " << ms << " ms: pushed "
                << root->pushed_.load() << ", flushed "
                << root->flushed_.load() << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
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

/** @return the next task id this thread would take (CreateTaskId's
 *  per-thread counter): the difference between two marks, minus one for the
 *  second mark itself, is the number of tasks this thread created between
 *  them. */
static clio::run::u32 TaskMark() { return clio::run::CreateTaskId().major_; }

/** Create the default cache (568.0) over the core. */
static void CreateMainCache() {
  CreateCache(kPool, "shm_wtc_main", 64ull << 20);
}

TEST_CASE("ShmWtc - the cache keys come from the CTE Create",
          "[cte][shm_wtc][keys]") {
  WtcFixture fx;
  CreateMainCache();
  // Get-or-create of the (existing) core pool: its OUT params carry the keys
  // of the cache fronting it on this node.
  clio::cte::core::Client probe;
  auto create = probe.AsyncCreate(clio::run::PoolQuery::Local(),
                                  std::string(clio::cte::core::kCtePoolName),
                                  clio::cte::core::kCtePoolId,
                                  clio::cte::core::CreateParams());
  create.Wait();
  REQUIRE(create->GetReturnCode() == 0);
  const clio::cte::core::ShmCacheKeys keys =
      clio::run::Task::Deserialize<clio::cte::core::CreateParams>(
          create->chimod_params_)
          .cache_keys_;
  REQUIRE(keys.Valid());
  REQUIRE(keys.cache_pool_ == kPool.ToU64());
  REQUIRE(keys.root_off_ == KeysOf(kPool).root_off_);

  // Connecting needs a client whose tasks pass the cache.
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  REQUIRE_FALSE(core.ConnectCache(keys));
  REQUIRE_FALSE(core.HasCache());
  clio::cte::core::Client io(kPool);
  REQUIRE(io.ConnectCache(keys));
  REQUIRE(io.HasCache());
}

TEST_CASE("ShmWtc - a connected PutBlob creates no task",
          "[cte][shm_wtc][paths]") {
  WtcFixture fx;
  CreateMainCache();
  clio::cte::core::Client io(kPool);
  Connect(io, kPool);
  const wtc::Stats *st = io.CacheStats();
  REQUIRE(st != nullptr);
  TagId tag = MakeTag(io, "wtc_no_task");
  const std::string val = Payload(64 * 1024, 'n');

  const clio::run::u64 puts0 = st->puts_.load();
  const clio::run::u32 m0 = TaskMark();
  REQUIRE(io.PutBlob(tag, "nt", 0, val.size(), val.data()) == 0);
  REQUIRE(TaskMark() == m0 + 1);  // no task between the marks
  REQUIRE(st->puts_.load() == puts0 + 1);

  // The bytes are readable from the segment, again with no task.
  std::string back(val.size(), '\0');
  const clio::run::u32 m1 = TaskMark();
  REQUIRE(io.TryGetCached(tag, "nt", 0, back.size(), back.data()));
  clio::run::u64 size = 0;
  REQUIRE(io.GetBlobSize(tag, "nt", &size) == 0);
  REQUIRE(TaskMark() == m1 + 1);
  REQUIRE(back == val);
  REQUIRE(size == val.size());

  // And the flusher carries them down the chain.
  REQUIRE(WaitFlushed(io));
  REQUIRE(CoreRead(tag, "nt", 0, val.size()) == val);
}

TEST_CASE("ShmWtc - without a cache, PutBlob/GetBlob take the task path",
          "[cte][shm_wtc][paths]") {
  WtcFixture fx;
  CreateMainCache();
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  REQUIRE_FALSE(core.HasCache());
  REQUIRE(core.CacheStats() == nullptr);
  TagId tag = MakeTag(core, "wtc_task_path");
  const std::string val = Payload(32 * 1024, 't');

  const clio::run::u32 m0 = TaskMark();
  REQUIRE(core.PutBlob(tag, "tp", 0, val.size(), val.data()) == 0);
  REQUIRE(TaskMark() > m0 + 1);  // a PutBlob task was created

  std::string back(val.size(), '\0');
  REQUIRE_FALSE(core.TryGetCached(tag, "tp", 0, back.size(), back.data()));
  REQUIRE(core.GetBlob(tag, "tp", 0, back.size(), back.data()) == 0);
  REQUIRE(back == val);
  clio::run::u64 size = 0;
  REQUIRE(core.GetBlobSize(tag, "tp", &size) == 0);
  REQUIRE(size == val.size());
  REQUIRE(CoreRead(tag, "tp", 0, val.size()) == val);
}

TEST_CASE("ShmWtc - writes and reads complete in shared memory",
          "[cte][shm_wtc]") {
  WtcFixture fx;
  CreateMainCache();
  clio::cte::core::Client io(kPool);
  Connect(io, kPool);
  const wtc::Stats *st = io.CacheStats();
  REQUIRE(st != nullptr);

  SECTION("put/get round trip without tasks, then write-through") {
    TagId tag = MakeTag(io, "wtc_roundtrip");
    const std::string val = Payload(256 * 1024, 'a');
    const clio::run::u64 puts0 = st->puts_.load();
    REQUIRE(io.PutBlob(tag, "b0", 0, val.size(), val.data()) == 0);
    REQUIRE(st->puts_.load() == puts0 + 1);

    // A brand-new blob of a brand-new tag is complete in the cache.
    std::string back(val.size(), '\0');
    const clio::run::u64 hits0 = st->get_hits_.load();
    REQUIRE(io.GetBlob(tag, "b0", 0, back.size(), back.data()) == 0);
    REQUIRE(back == val);
    REQUIRE(st->get_hits_.load() == hits0 + 1);

    clio::run::u64 size = 0;
    REQUIRE(io.GetBlobSize(tag, "b0", &size) == 0);
    REQUIRE(size == val.size());

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
    REQUIRE(io.PutBlob(tag, "r", 0, a.size(), a.data()) == 0);
    REQUIRE(io.PutBlob(tag, "r", 64 * 1024, b.size(), b.data()) == 0);
    REQUIRE(WaitFlushed(io));

    std::string want = base;
    std::memcpy(&want[0], a.data(), a.size());
    std::memcpy(&want[64 * 1024], b.data(), b.size());
    REQUIRE(CoreRead(tag, "r", 0, want.size()) == want);

    // Reads of a partially cached blob go to the chain (and fill).
    std::string back(want.size(), '\0');
    REQUIRE(io.GetBlob(tag, "r", 0, back.size(), back.data()) == 0);
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
    REQUIRE(io.PutBlob(tag, "v", 4096, a.size(), a.data()) == 0);

    std::string back(4096, '\0');
    const clio::run::u64 hits0 = st->get_hits_.load();
    const clio::run::u64 fills0 = st->fills_.load();
    REQUIRE(io.GetBlob(tag, "v", 8192, back.size(), back.data()) == 0);
    REQUIRE(back == std::string(4096, 'A'));
    REQUIRE(st->get_hits_.load() == hits0 + 1);  // served from the segment
    REQUIRE(st->fills_.load() == fills0);         // no trip to the chain

    // A read touching bytes the cache never saw is the chain's (not zeros).
    std::string span(8192, '\0');
    REQUIRE(io.GetBlob(tag, "v", 0, span.size(), span.data()) == 0);
    REQUIRE(span == std::string(4096, 'x') + std::string(4096, 'A'));

    // Another writer's invalidation: the written range no longer serves.
    const std::string c(8192, 'C');
    REQUIRE(io.PutBlob(tag, "v", 4096, c.size(), c.data()) == 0);
    REQUIRE(WaitFlushed(io));
    TestView(io.CacheKeys())->Invalidate(tag.major_, tag.minor_, "v", 1);
    core.AsyncPutBlob(tag, "v", 4096, a.size(), a.data()).Wait();
    REQUIRE(io.GetBlob(tag, "v", 8192, back.size(), back.data()) == 0);
    REQUIRE(back == std::string(4096, 'A'));  // the chain's newer bytes
  }

  SECTION("small writes coalesce into few batches") {
    // A cache of its own whose flusher never runs on its period during the
    // writes (60 s): every write queues first, then one explicit flush pass
    // (FlushLog, which drains the log) lands them. With the default 50 us
    // period a slow runner flushes each write as it arrives (one batch per
    // write), and the flusher's pins refuse appends busy, so neither the
    // batch count nor the range count would depend on the code alone.
    const clio::run::PoolId cpool(579, 0);
    CreateCache(cpool, "shm_wtc_coalesce", 64ull << 20, false, 0,
                /*flush_period_us=*/60e6);
    clio::cte::core::Client cio(cpool);
    Connect(cio, cpool);
    const wtc::Stats *cst = cio.CacheStats();
    TagId tag = MakeTag(cio, "wtc_coalesce");
    std::string want(128 * 512, '\0');
    const clio::run::u64 batches0 = cst->flush_batches_.load();
    const clio::run::u64 ranges0 = cst->flush_ranges_.load();
    const clio::run::u64 puts0 = cst->puts_.load();
    for (int i = 0; i < 128; ++i) {
      std::string piece(512, static_cast<char>('a' + (i % 26)));
      std::memcpy(&want[i * 512], piece.data(), piece.size());
      REQUIRE(cio.PutBlob(tag, "seq", i * 512, piece.size(), piece.data()) == 0);
    }
    REQUIRE(cst->puts_.load() - puts0 == 128);  // all cached, none flushed
    clio::cte::shm_wtc::Client flusher(cpool);
    auto flush = flusher.AsyncFlushLog();
    flush.Wait();
    REQUIRE(flush->GetReturnCode() == 0);
    REQUIRE(WaitFlushed(cio));
    // One flushed range per cached write, carried by far fewer batches.
    const clio::run::u64 ranges = cst->flush_ranges_.load() - ranges0;
    const clio::run::u64 batches = cst->flush_batches_.load() - batches0;
    INFO("flushed ranges " << ranges << ", batches " << batches);
    REQUIRE(ranges == 128);
    REQUIRE(batches < 64);
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
    REQUIRE(io.GetBlob(tag, "cold", 0, back.size(), back.data()) == 0);
    REQUIRE(back == val);
    REQUIRE(st->fills_.load() == fills0 + 1);

    const clio::run::u64 hits0 = st->get_hits_.load();
    std::string back2(4096, '\0');
    REQUIRE(io.GetBlob(tag, "cold", 8192, back2.size(), back2.data()) == 0);
    REQUIRE(back2 == val.substr(8192, 4096));
    REQUIRE(st->get_hits_.load() == hits0 + 1);
  }

  SECTION("a non-plain put takes the task path and stays ordered") {
    TagId tag = MakeTag(io, "wtc_taskpath");
    const std::string v1(8192, '1');
    const std::string v2(8192, '2');
    REQUIRE(io.PutBlob(tag, "t", 0, v1.size(), v1.data()) == 0);
    // An explicit score is not expressible in the cache: task path, after
    // the buffered write lands.
    REQUIRE(io.PutBlob(tag, "t", 0, v2.size(), v2.data(), 0.75f) == 0);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "t", 0, v2.size()) == v2);
    std::string back(v2.size(), '\0');
    REQUIRE(io.GetBlob(tag, "t", 0, back.size(), back.data()) == 0);
    REQUIRE(back == v2);
  }

  SECTION("DelBlob removes the cached copy and the chain's") {
    TagId tag = MakeTag(io, "wtc_del");
    const std::string v(4096, 'd');
    REQUIRE(io.PutBlob(tag, "gone", 0, v.size(), v.data()) == 0);
    auto d = io.AsyncDelBlob(tag, "gone");
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
    clio::run::u64 size = 0;
    REQUIRE(io.GetBlobSize(tag, "gone", &size) != 0);
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
          const clio::run::u32 p_rc = io.PutBlob(tag, name, 0, v.size(), v.data());
          std::string back(v.size(), '\0');
          const clio::run::u32 g_rc = io.GetBlob(tag, name, 0, back.size(), back.data());
          if (p_rc != 0 || g_rc != 0 || back != v) {
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
  Connect(io, pool);
  const wtc::Stats *st = io.CacheStats();
  TagId tag = MakeTag(io, "wtc_evict");

  // 48 MB through an 8 MB cache: only possible if clean buffers are evicted.
  std::vector<std::string> vals;
  for (int i = 0; i < 48; ++i) {
    vals.push_back(Payload(1 << 20, static_cast<char>('A' + (i % 26))));
    REQUIRE(io.PutBlob(tag, "e" + std::to_string(i), 0, vals.back().size(), vals.back().data()) == 0);
  }
  REQUIRE(WaitFlushed(io));
  REQUIRE(st->evictions_.load() > 0);
  for (int i = 0; i < 48; ++i) {
    std::string back(1 << 20, '\0');
    REQUIRE(io.GetBlob(tag, "e" + std::to_string(i), 0, back.size(), back.data()) == 0);
    REQUIRE(back == vals[i]);
  }
}

TEST_CASE("ShmWtc - overwrites flush once and the last write wins",
          "[cte][shm_wtc][dedup]") {
  WtcFixture fx;
  // A range is retired without a resend only when it is still in the log
  // after an EARLIER batch copied its blob: overwrites popped into the same
  // batch coalesce into one record and count as nothing. So the pile-up
  // must outgrow one batch. With the default 16 MiB batch (256 of these
  // 64 KiB puts) a slow runner (icx on windows-2022) put fewer than that
  // per 100 ms flusher period, every batch took the whole log, and nothing
  // was ever skipped. A one-put batch makes any two queued overwrites of a
  // blob a skip, however fast or slow the machine; the slow flusher (100 ms)
  // still lets them queue instead of racing a 50 us one.
  const clio::run::PoolId kDedupPool(576, 0);
  CreateCache(kDedupPool, "shm_wtc_dedup", 64ull << 20, false, 0, 100000.0,
              clio::cte::core::kCtePoolId, 64ull << 10);
  clio::cte::core::Client io(kDedupPool);
  Connect(io, kDedupPool);
  const wtc::Stats *st = io.CacheStats();
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
      REQUIRE(io.PutBlob(tag, "d" + std::to_string(b), 0, last[b].size(), last[b].data()) == 0);
    }
  }
  // A partial overwrite after all that must not be skipped.
  std::string tail = Payload(4096, 'Z');
  {
    REQUIRE(io.PutBlob(tag, "d0", 8192, tail.size(), tail.data()) == 0);
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
  clio::run::u64 size = 0;
  REQUIRE(io.GetBlobSize(tag, name, &size) == 0);
  return size;
}

TEST_CASE("ShmWtc - md_only caches blob sizes and drops them on change",
          "[cte][shm_wtc][md_only]") {
  WtcFixture fx;
  const clio::run::PoolId pool(571, 0);
  CreateCache(pool, "shm_wtc_md", 32ull << 20, /*md_only=*/true);
  clio::cte::core::Client io(pool);
  Connect(io, pool);
  const wtc::Stats *st = io.CacheStats();
  TagId tag = MakeTag(io, "wtc_md");

  // Bytes are not cached: a put takes the task path and lands in the core.
  const std::string v1 = Payload(1000, 'a');
  REQUIRE(io.PutBlob(tag, "m", 0, v1.size(), v1.data()) == 0);
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
  REQUIRE(io.PutBlob(tag, "m", 0, v2.size(), v2.data()) == 0);
  REQUIRE(SizeOf(io, tag, "m") == 3000);
  auto t = io.AsyncTruncateBlob(tag, "m", 700);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  REQUIRE(SizeOf(io, tag, "m") == 700);
  std::string back(700, '\0');
  REQUIRE(io.GetBlob(tag, "m", 0, back.size(), back.data()) == 0);
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
  const wtc::Stats *st = io.CacheStats();
  TagId tag = MakeTag(io, tname);
  auto put = [&](const std::string &name, size_t n, char c) {
    const std::string v = Payload(n, c);
    REQUIRE(io.PutBlob(tag, name, 0, v.size(), v.data()) == 0);
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
  TagId o2;
  clio::run::u64 o2_size = 0;
  REQUIRE(io.GetOrCreateTag(tname, &o2, &o2_size) == 0);
  REQUIRE(o2 == tag);
  REQUIRE(o2_size == 600);
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
  Connect(io, kPool);
  TagMetaScenario(io, "wtc_tagmeta_data");
}

TEST_CASE("ShmWtc - tag attributes and opens are cached and exact (md_only)",
          "[cte][shm_wtc][tagmeta][md_only]") {
  WtcFixture fx;
  const clio::run::PoolId pool(572, 0);
  CreateCache(pool, "shm_wtc_md2", 32ull << 20, /*md_only=*/true);
  clio::cte::core::Client io(pool);
  Connect(io, pool);
  TagMetaScenario(io, "wtc_tagmeta_md");
}

TEST_CASE("ShmWtc - truncate through the cache drops cached bytes",
          "[cte][shm_wtc][truncate]") {
  WtcFixture fx;
  clio::cte::core::Client io(kPool);
  Connect(io, kPool);
  TagId tag = MakeTag(io, "wtc_trunc");
  const std::string v = Payload(8192, 'q');
  REQUIRE(io.PutBlob(tag, "t", 0, v.size(), v.data()) == 0);
  REQUIRE(SizeOf(io, tag, "t") == 8192);  // cached with its bytes
  auto t = io.AsyncTruncateBlob(tag, "t", 100);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  REQUIRE(SizeOf(io, tag, "t") == 100);
  std::string back(100, '\0');
  REQUIRE(io.GetBlob(tag, "t", 0, back.size(), back.data()) == 0);
  REQUIRE(back == v.substr(0, 100));
}

/**
 * The cache's put-refusal counters, for an INFO: when a put the test expects
 * cached is written around, they say why it was refused.
 * @param st the cache's stats
 * @return "full_buf=.. full_slot=.. full_log=.. miss=.. busy=.. waits=..
 *         evictions=.."
 */
static std::string RefusalCounters(const wtc::Stats *st) {
  return "full_buf=" + std::to_string(st->put_full_buf_.load()) +
         " full_slot=" + std::to_string(st->put_full_slot_.load()) +
         " full_log=" + std::to_string(st->put_full_log_.load()) +
         " miss=" + std::to_string(st->put_miss_.load()) +
         " busy=" + std::to_string(st->put_busy_.load()) +
         " waits=" + std::to_string(st->waits_.load()) +
         " evictions=" + std::to_string(st->evictions_.load());
}

TEST_CASE("ShmWtc - large puts displace a cache full of small buffers",
          "[cte][shm_wtc][evict][fragment]") {
  WtcFixture fx;
  const clio::run::PoolId pool(570, 0);
  CreateCache(pool, "shm_wtc_frag", 8ull << 20);  // 8 MB segment
  clio::cte::core::Client io(pool);
  Connect(io, pool);
  const wtc::Stats *st = io.CacheStats();
  TagId tag = MakeTag(io, "wtc_frag");

  // Fill the segment with clean 4 KB buffers: freeing any one of them never
  // leaves room for a 1 MB buffer, since the allocator does not merge on
  // free.
  const std::string small = Payload(4096, 's');
  for (int i = 0; i < 4000; ++i) {
    REQUIRE(io.PutBlob(tag, "s" + std::to_string(i), 0, small.size(), small.data()) == 0);
  }
  REQUIRE(WaitFlushed(io));

  // Every large put must land in the cache (puts_ counts cached writes
  // only; a write-around bypasses it). The time is reported, not bounded:
  // a regression that stops displacing small buffers writes around, which
  // puts_ catches; one that hangs trips the ctest TIMEOUT.
  const clio::run::u64 puts0 = st->puts_.load();
  const std::string before = RefusalCounters(st);
  std::vector<std::string> vals;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 4; ++i) {
    vals.push_back(Payload(1 << 20, static_cast<char>('a' + i)));
    REQUIRE(io.PutBlob(tag, "L" + std::to_string(i), 0, vals.back().size(), vals.back().data()) == 0);
  }
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  INFO("large puts took " << ms << " ms; before: " << before
       << "; after: " << RefusalCounters(st));
  REQUIRE(st->puts_.load() - puts0 == 4);
  REQUIRE(WaitFlushed(io));
  for (int i = 0; i < 4; ++i) {
    std::string back(1 << 20, '\0');
    REQUIRE(io.GetBlob(tag, "L" + std::to_string(i), 0, back.size(), back.data()) == 0);
    REQUIRE(back == vals[i]);
    REQUIRE(CoreRead(tag, "L" + std::to_string(i), 0, back.size()) ==
            vals[i]);
  }
}

TEST_CASE("ShmWtc - a put refused for room waits for the flush that makes it",
          "[cte][shm_wtc][evict][flushwait]") {
  // More dirty bytes than the heap holds: later puts fit only once earlier
  // ones are flushed (clean, evictable). Cache-path writes do not wake the
  // flusher, and an idle runtime's worker parks for up to max_sleep
  // (50 ms), so a refused put must wake the flusher and wait for the
  // ranges buffered before it, not write around after ~1 ms.
  WtcFixture fx;
  const clio::run::PoolId pool(578, 0);
  CreateCache(pool, "shm_wtc_flushwait", 8ull << 20);  // ~5 MB of heap
  clio::cte::core::Client io(pool);
  Connect(io, pool);
  const wtc::Stats *st = io.CacheStats();
  TagId tag = MakeTag(io, "wtc_flushwait");

  // Idle long enough for the flusher's worker to back off its park: sets
  // up the parked-worker case. The result does not depend on it: the
  // cache waits up to kTestNoSpaceWaitMs for the flush either way.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  constexpr int kPuts = 8;  // 8 MB of 1 MB puts
  const clio::run::u64 puts0 = st->puts_.load();
  std::vector<std::string> vals;
  for (int i = 0; i < kPuts; ++i) {
    vals.push_back(Payload(1 << 20, static_cast<char>('A' + i)));
    REQUIRE(io.PutBlob(tag, "W" + std::to_string(i), 0, vals.back().size(),
                       vals.back().data()) == 0);
  }
  INFO("cached puts " << st->puts_.load() - puts0 << " of " << kPuts << "; "
       << RefusalCounters(st));
  REQUIRE(st->puts_.load() - puts0 == static_cast<clio::run::u64>(kPuts));
  REQUIRE(WaitFlushed(io));
  for (int i = 0; i < kPuts; ++i) {
    REQUIRE(CoreRead(tag, "W" + std::to_string(i), 0, vals[i].size()) ==
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
  Connect(io, kExtPool);
  const wtc::Stats *st = io.CacheStats();
  REQUIRE(st != nullptr);

  SECTION("sizes that are not powers of two round trip and flush") {
    TagId tag = MakeTag(io, "ext_sizes");
    const std::vector<size_t> sizes = {1, 4096, 4097, 100 * 1024,
                                       (1u << 20) + 1};
    for (size_t i = 0; i < sizes.size(); ++i) {
      const std::string name = "s" + std::to_string(i);
      const std::string v = Payload(sizes[i], static_cast<char>('a' + i));
      REQUIRE(io.PutBlob(tag, name, 0, v.size(), v.data()) == 0);
      std::string back(v.size(), '\0');
      const clio::run::u64 hits0 = st->get_hits_.load();
      REQUIRE(io.GetBlob(tag, name, 0, back.size(), back.data()) == 0);
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
      REQUIRE(io.PutBlob(tag, "g", want.size(), piece.size(), piece.data()) == 0);
      want += piece;
    }
    std::string back(want.size(), '\0');
    REQUIRE(io.GetBlob(tag, "g", 0, back.size(), back.data()) == 0);
    REQUIRE(back == want);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "g", 0, want.size()) == want);
  }

  SECTION("a working set larger than the lease evicts and stays correct") {
    TagId tag = MakeTag(io, "ext_evict");
    const int kBlobs = 40;  // 40 MiB through a 16 MiB lease
    for (int k = 0; k < kBlobs; ++k) {
      const std::string v = Payload(1u << 20, static_cast<char>(k));
      REQUIRE(io.PutBlob(tag, "e" + std::to_string(k), 0, v.size(), v.data()) == 0);
    }
    REQUIRE(WaitFlushed(io));
    for (int k = 0; k < kBlobs; ++k) {
      const std::string v = Payload(1u << 20, static_cast<char>(k));
      std::string back(v.size(), '\0');
      const clio::run::u64 hits0 = st->get_hits_.load();
      const clio::run::u64 fills0 = st->fills_.load();
      REQUIRE(io.GetBlob(tag, "e" + std::to_string(k), 0, back.size(), back.data()) == 0);
      if (back != v) {
        // Where the bytes came from (a cache hit, a fill, the chain) and
        // whether the chain itself holds the right ones.
        const std::string below =
            CoreRead(tag, "e" + std::to_string(k), 0, v.size());
        INFO("blob e" << k << " read wrong bytes: " << DescribeMismatch(v, back)
             << "; cache hits +" << st->get_hits_.load() - hits0 << " fills +"
             << st->fills_.load() - fills0 << "; chain copy "
             << (below == v ? "correct" : DescribeMismatch(v, below)));
      }
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
      REQUIRE(io.PutBlob(tag, "s" + std::to_string(i), 0, small.size(), small.data()) == 0);
    }
    REQUIRE(WaitFlushed(io));
    const clio::run::u64 puts0 = st->puts_.load();
    const std::string before = RefusalCounters(st);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; ++i) {
      const std::string v = Payload(1 << 20, static_cast<char>('a' + i));
      REQUIRE(io.PutBlob(tag, "L" + std::to_string(i), 0, v.size(), v.data()) == 0);
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    INFO("large puts took " << ms << " ms; before: " << before
         << "; after: " << RefusalCounters(st));
    // Cached, not written around (the time is reported, not bounded).
    REQUIRE(st->puts_.load() - puts0 == 4);
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
    REQUIRE(io.PutBlob(tag, "p", 70000, a.size(), a.data()) == 0);
    std::string want = base;
    std::memcpy(&want[70000], a.data(), a.size());
    std::string back(want.size(), '\0');
    REQUIRE(io.GetBlob(tag, "p", 0, back.size(), back.data()) == 0);
    REQUIRE(back == want);
    REQUIRE(WaitFlushed(io));
    REQUIRE(CoreRead(tag, "p", 0, want.size()) == want);
  }
}

/**
 * Format a private segment for a View-only test (no runtime involved).
 * @param view the view to format
 * @param tag  distinguishes this test's segment name
 */
static void FormatPrivateView(wtc::View &view, const std::string &tag) {
  const clio::run::u32 pid = static_cast<clio::run::u32>(getpid());
  wtc::Limits lim;
  lim.seg_size_ = 16ull << 20;
  lim.max_blobs_ = 1024;
  lim.log_depth_ = 1024;
  const std::string name = "clio_wtc_viewtest_" + tag + "_" +
                           std::to_string(pid);
  const ctp::ipc::MemoryBackendId id(pid, 0x57550000u);
  REQUIRE(view.Format(name, id, lim, /*node_id=*/0));
  REQUIRE(view.Ready());
}

/**
 * Write a blob through a View and take its slot off the dirty log, as the
 * flusher does (the range stays pending until CompleteRanges).
 * @return the blob's slot
 */
static clio::run::u32 PutAndPopSlot(wtc::View &view, const std::string &blob,
                                    const std::string &bytes) {
  REQUIRE(view.TryPut(1, 1, blob.data(), blob.size(), 0, bytes.size(),
                      bytes.data(), 0, /*may_block=*/false) ==
          wtc::Status::kOk);
  wtc::DirtyRange r{};
  REQUIRE(view.Log()->Pop(r));
  return r.slot_;
}

/** Milliseconds since `t0`. */
static double MsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

/** How long the pin tests wait for a step that must not block before they
 *  call it blocked. Generous (thread start-up on a loaded runner counts
 *  against it): a step that really waits for the pin never finishes, so
 *  any finite bound catches it; this one only has to never be hit by a
 *  step that is merely slow. */
static constexpr double kPinPromptMs = 30000.0;

/**
 * Poll `done` until it is true or kPinPromptMs passes.
 * @param done flag the other thread sets when its step returns
 * @return whether the step returned in time
 */
static bool WaitPrompt(const std::atomic<bool> &done) {
  const auto t0 = std::chrono::steady_clock::now();
  while (!done.load() && MsSince(t0) < kPinPromptMs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return done.load();
}

TEST_CASE("ShmWtc - a pinned entry refuses growth (kBusy) and never blocks "
          "the stripe lock",
          "[cte][shm_wtc][pin][deadlock]") {
  // The flusher pins a blob's bytes (zero-copy flush) until the home acks;
  // the home acks only after invalidating this node, which needs the
  // blob's stripe lock. A writer that grows the pinned blob must therefore
  // refuse (kBusy) instead of waiting for the pin under that lock.
  wtc::View view;
  FormatPrivateView(view, "busy");
  const wtc::Stats &st = view.GetRoot()->stats_;
  const std::string blob = "pinned";
  const std::string first = Payload(4096, 'p');
  const clio::run::u32 slot = PutAndPopSlot(view, blob, first);

  // The flush in flight: pin the logged range (pending stays 1).
  clio::run::u64 size = first.size();
  clio::run::u64 gen = 0;
  REQUIRE(view.PinRange(slot, 0, &size, &gen) != nullptr);

  // A write past the buffer's capacity (1 MiB + 47008 B, ior-hard style)
  // must grow, i.e. move, the pinned buffer. The writer retries WITHOUT
  // holding anything, as the callers do, until the attempt after the unpin:
  // it ends on an event, not on a clock.
  constexpr clio::run::u64 kGrowOff = 1ull << 20;
  const std::string grow = Payload(47008, 'g');
  std::atomic<int> first_status{-1};
  std::atomic<bool> first_done{false};
  std::atomic<int> final_status{-1};
  std::atomic<bool> writer_done{false};
  std::atomic<bool> unpinned{false};
  auto unpin = [&] {
    view.UnpinRange(slot);
    unpinned.store(true);
  };
  std::thread writer([&] {
    wtc::Status s = view.TryPut(1, 1, blob.data(), blob.size(), kGrowOff,
                                grow.size(), grow.data(), 0, false);
    first_status.store(static_cast<int>(s));
    first_done.store(true);
    while (s == wtc::Status::kBusy) {
      const bool after_unpin = unpinned.load();
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      s = view.TryPut(1, 1, blob.data(), blob.size(), kGrowOff, grow.size(),
                      grow.data(), 0, false);
      if (after_unpin) break;  // one try after the unpin decides
    }
    final_status.store(static_cast<int>(s));
    writer_done.store(true);
  });

  // The first refusal arrives (a regressed writer blocked on the pin is
  // unblocked and joined before the failure is reported).
  const bool prompt = WaitPrompt(first_done);
  if (!prompt) {
    unpin();
    writer.join();
  }
  REQUIRE(prompt);
  REQUIRE(first_status.load() == static_cast<int>(wtc::Status::kBusy));
  REQUIRE(st.put_busy_.load() >= 1);

  // The home's invalidation of this node, on the same stripe, completes
  // while the writer is still being refused.
  const clio::run::u64 inv0 = st.invalidations_.load();
  std::atomic<bool> inv_done{false};
  std::thread inval([&] {
    view.Invalidate(1, 1, blob.data(), blob.size());
    inv_done.store(true);
  });
  const bool inv_prompt = WaitPrompt(inv_done);
  const bool still_refused = !writer_done.load();
  if (!inv_prompt || !still_refused) {
    unpin();
    inval.join();
    writer.join();
  }
  REQUIRE(inv_prompt);
  REQUIRE(still_refused);  // still pinned: still refused
  inval.join();
  REQUIRE(st.invalidations_.load() == inv0 + 1);

  // The flush lands: unpin. The writer's next retry succeeds.
  unpin();
  writer.join();
  REQUIRE(final_status.load() == static_cast<int>(wtc::Status::kOk));
  std::string back(grow.size(), '\0');
  REQUIRE(view.TryGet(1, 1, blob.data(), blob.size(), kGrowOff, back.size(),
                      back.data()) == wtc::Status::kOk);
  REQUIRE(back == grow);
  view.CompleteRanges(slot, 1);
}

TEST_CASE("ShmWtc - invalidate and drop of a pinned, flushed entry return "
          "at once",
          "[cte][shm_wtc][pin][deadlock]") {
  // pending_ is 0 (the range completed) but a pin is still held: Invalidate
  // and DropBlob must not wait for it under the stripe lock. Run on one
  // thread: a wait here would never end.
  wtc::View view;
  FormatPrivateView(view, "drop");
  const std::string bytes = Payload(8192, 'd');
  for (const std::string blob : {"inv", "del"}) {
    const clio::run::u32 slot = PutAndPopSlot(view, blob, bytes);
    view.CompleteRanges(slot, 1);
    clio::run::u64 size = bytes.size();
    clio::run::u64 gen = 0;
    REQUIRE(view.PinRange(slot, 0, &size, &gen) != nullptr);
    // Single-threaded: a call that waited for the pin would never return,
    // so a regression hangs here and the ctest TIMEOUT reports it; no
    // wall-clock bound is needed.
    if (blob == "inv") {
      view.Invalidate(1, 1, blob.data(), blob.size());
    } else {
      view.DropBlob(1, 1, blob.data(), blob.size());
    }
    std::string back(bytes.size(), '\0');
    // Nothing stale is served, and no size is known.
    REQUIRE(view.TryGet(1, 1, blob.data(), blob.size(), 0, back.size(),
                        back.data()) == wtc::Status::kMiss);
    clio::run::u64 sz = 0;
    REQUIRE(view.TryGetSize(1, 1, blob.data(), blob.size(), &sz) ==
            wtc::Status::kMiss);
    view.UnpinRange(slot);
    // The entry is writable again afterwards.
    REQUIRE(view.TryPut(1, 1, blob.data(), blob.size(), 0, bytes.size(),
                        bytes.data(), 0, false) == wtc::Status::kOk);
    REQUIRE(view.TryGet(1, 1, blob.data(), blob.size(), 0, back.size(),
                        back.data()) == wtc::Status::kOk);
    REQUIRE(back == bytes);
  }
}

/** The core over one file device (WtcFixture) and a cache over it. */
static const clio::run::PoolId kFullCorePool(513, 0);
static const clio::run::PoolId kFullCachePool(577, 0);

/**
 * Make the full-device core's file target refuse every write with ENOSPC
 * (the file bdev's TEST-ONLY `.nospace` marker), or work again.
 * @param full true: the device is full; false: it has room again
 */
static void SetDeviceFull(bool full) {
  auto *ipc = CLIO_IPC;
  // The core registers a device on node N as `<path>_node<N>`.
  const std::string marker = DataDir() + "/" + WtcFixture::kFullDeviceFile +
                             "_node" + std::to_string(ipc->GetNodeId()) +
                             ".nospace";
  if (full) {
    std::ofstream(marker).put('x');
  } else {
    fs::remove(marker);
  }
  // Longer than the file bdev's 100 ms marker re-check period.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

TEST_CASE("ShmWtc - a device out of space is ENOSPC, not an I/O error",
          "[cte][shm_wtc][nospace]") {
  // A full persistent device (a DAOS pool with no room left) used to fail
  // PutBlob with rc 21, a generic write error: clio-fs reported EIO from
  // write(2) and fsync(2). It must be the out-of-space code, so the
  // filesystem returns ENOSPC.
  WtcFixture fx;
  CreateCache(kFullCachePool, "shm_wtc_full", 64ull << 20, false, 0, 0,
              kFullCorePool);
  const std::string val = Payload(64 * 1024, 'f');

  // Task path, straight to the core: the device refuses the bytes.
  clio::cte::core::Client core(kFullCorePool);
  TagId tag = MakeTag(core, "wtc_full_task");
  REQUIRE(core.PutBlob(tag, "p", 0, val.size(), val.data()) == 0);
  SetDeviceFull(true);
  for (clio::run::u64 off : {clio::run::u64(0), clio::run::u64(val.size())}) {
    const clio::run::u32 rc =
        core.PutBlob(tag, "p", off, val.size(), val.data());
    INFO("PutBlob at " << off << " on a full device: rc " << rc);
    REQUIRE(rc == clio::cte::core::kPutDeviceNoSpaceRc);
    REQUIRE(clio::cte::core::PutRcIsNoSpace(rc));
  }

  // Cache path: the write is absorbed, its flush fails, and the fsync of
  // the tag reports the full device.
  clio::cte::core::Client io(kFullCachePool);
  Connect(io, kFullCachePool);
  TagId ctag = MakeTag(io, "wtc_full_cached");
  REQUIRE(io.PutBlob(ctag, "c", 0, val.size(), val.data()) == 0);
  REQUIRE(WaitFlushed(io));
  auto sync = io.AsyncSyncTag(ctag, clio::run::PoolQuery::Local());
  sync.Wait();
  INFO("SyncTag after a failed flush: rc " << sync->GetReturnCode());
  REQUIRE(sync->GetReturnCode() == clio::cte::core::kSyncNoSpaceRc);

  // Room again: writes land.
  SetDeviceFull(false);
  REQUIRE(core.PutBlob(tag, "p", val.size(), val.size(), val.data()) == 0);
  REQUIRE(io.PutBlob(ctag, "c", 0, val.size(), val.data()) == 0);
  REQUIRE(WaitFlushed(io));
  auto sync2 = io.AsyncSyncTag(ctag, clio::run::PoolQuery::Local());
  sync2.Wait();
  REQUIRE(sync2->GetReturnCode() == 0);
  std::string back(val.size(), '\0');
  auto get = core.AsyncGetBlob(ctag, "c", 0, back.size(), 0, back.data());
  get.Wait();
  REQUIRE(get->GetReturnCode() == 0);
  REQUIRE(back == val);
}

// ===========================================================================
// Extent-mode stress: concurrent writers, readers, evictions and fills
// ===========================================================================

/** Stress: writer threads; each owns kStressBlobsPerThread blobs. */
static constexpr int kStressThreads = 4;
/** Stress: blobs each thread owns (writes and reads exactly). */
static constexpr int kStressBlobsPerThread = 8;
/** Stress: total blobs. */
static constexpr int kStressBlobs = kStressThreads * kStressBlobsPerThread;
/** Stress: largest blob (bytes); writes never grow a blob past it. */
static constexpr clio::run::u64 kStressMaxBlob = 640ull << 10;
/** Stress: default run time (ms); CLIO_WTC_STRESS_MS overrides. */
static constexpr int kStressDefaultMs = 4000;
/** Stress: most mismatch reports kept for the failure message. */
static constexpr size_t kStressMaxReports = 8;

/**
 * One 8-byte stress word: the blob, the write that produced it and the
 * word's own position, so any read can tell whose bytes it got.
 * @param blob blob id (1-based; 0 never written: a hole reads as 0)
 * @param ver  write number of that blob
 * @param idx  word index inside the blob (byte offset / 8)
 * @return the word
 */
static clio::run::u64 StressWord(clio::run::u32 blob, clio::run::u32 ver,
                                 clio::run::u64 idx) {
  return (static_cast<clio::run::u64>(blob & 0xFFFFu) << 48) |
         (static_cast<clio::run::u64>(ver & 0xFFFFu) << 32) |
         (idx & 0xFFFFFFFFull);
}

/**
 * Bytes of write `ver` of `blob` over [off, off + size).
 * @param blob blob id
 * @param ver  write number
 * @param off  byte offset in the blob (multiple of 8)
 * @param size bytes (multiple of 8)
 * @return the bytes
 */
static std::string StressBytes(clio::run::u32 blob, clio::run::u32 ver,
                               clio::run::u64 off, clio::run::u64 size) {
  std::string s(size, '\0');
  for (clio::run::u64 i = 0; i < size; i += 8) {
    const clio::run::u64 w = StressWord(blob, ver, (off + i) / 8);
    std::memcpy(&s[i], &w, 8);
  }
  return s;
}

/**
 * Decode one stress word for a failure message.
 * @param s   buffer
 * @param pos byte offset of the word in s
 * @return "blob B ver V at byte X" or "zero"
 */
static std::string StressWordAt(const std::string &s, size_t pos) {
  clio::run::u64 w = 0;
  std::memcpy(&w, s.data() + pos, 8);
  if (w == 0) return "zero";
  std::ostringstream os;
  os << "blob " << (w >> 48) << " ver " << ((w >> 32) & 0xFFFFu)
     << " at byte " << (w & 0xFFFFFFFFull) * 8;
  return os.str();
}

/**
 * Compare an exact read of a blob range with the owner's model.
 * @param want the model's bytes of the range
 * @param got  the bytes read
 * @param off  byte offset of the range in the blob
 * @return "" when equal; else the first differing word decoded both ways
 *         and how many words differ
 */
static std::string StressDiff(const std::string &want, const std::string &got,
                              clio::run::u64 off) {
  if (want == got) return "";
  size_t first = want.size();
  size_t nbad = 0;
  for (size_t i = 0; i + 8 <= want.size(); i += 8) {
    if (std::memcmp(want.data() + i, got.data() + i, 8) == 0) continue;
    if (first == want.size()) first = i;
    ++nbad;
  }
  if (first == want.size()) return "sizes differ";
  std::ostringstream os;
  os << nbad << " words differ; first at byte " << off + first
     << " (granule " << (off + first) / 4096 << "): want "
     << StressWordAt(want, first) << ", got " << StressWordAt(got, first);
  return os.str();
}

/**
 * Check a read of a blob another thread writes concurrently: each word must
 * be a hole (0) or a word of THAT blob written at THAT position (any write).
 * @param got  the bytes read
 * @param blob the blob id read
 * @param off  byte offset of the read in the blob
 * @return "" when consistent, else the first foreign word decoded
 */
static std::string StressForeignCheck(const std::string &got,
                                      clio::run::u32 blob,
                                      clio::run::u64 off) {
  for (size_t i = 0; i + 8 <= got.size(); i += 8) {
    clio::run::u64 w = 0;
    std::memcpy(&w, got.data() + i, 8);
    if (w == 0) continue;
    if ((w >> 48) == (blob & 0xFFFFu) &&
        (w & 0xFFFFFFFFull) == (off + i) / 8) {
      continue;
    }
    std::ostringstream os;
    os << "byte " << off + i << " (granule " << (off + i) / 4096
       << ") holds " << StressWordAt(got, i);
    return os.str();
  }
  return "";
}

/** State the stress threads share. */
struct StressShared {
  clio::cte::core::Client *io_ = nullptr;
  TagId tag_;
  /** Size of each blob as of its owner's last COMPLETED write (only grows):
   *  a concurrent reader stays inside it. */
  std::atomic<clio::run::u64> sizes_[kStressBlobs];
  std::atomic<bool> stop_{false};
  std::atomic<clio::run::u64> writes_{0};
  std::atomic<clio::run::u64> reads_{0};
  std::atomic<clio::run::u64> bad_{0};
  std::mutex mu_;
  std::vector<std::string> reports_;

  StressShared() {
    for (auto &s : sizes_) s.store(0);
  }

  /**
   * Record one wrong result (kept up to kStressMaxReports).
   * @param what description
   */
  void Report(const std::string &what) {
    bad_.fetch_add(1);
    std::lock_guard<std::mutex> g(mu_);
    if (reports_.size() < kStressMaxReports) reports_.push_back(what);
  }
};

/** Name of stress blob `id`. */
static std::string StressName(clio::run::u32 id) {
  return "x" + std::to_string(id);
}

/** One owner thread's blobs: exact contents and write counters. */
struct StressModel {
  std::vector<std::string> bytes_;
  std::vector<clio::run::u32> vers_;
};

/**
 * Stress write: a random range of one owned blob (a rewrite, an append or a
 * growth past a hole), of a random size class, then update the model.
 * @param sh    shared state
 * @param rng   this thread's generator
 * @param m     this thread's model
 * @param first id of this thread's first blob
 */
static void StressWrite(StressShared &sh, std::mt19937_64 &rng, StressModel &m,
                        clio::run::u32 first) {
  const size_t j = rng() % kStressBlobsPerThread;
  const clio::run::u32 id = first + static_cast<clio::run::u32>(j);
  std::string &model = m.bytes_[j];
  clio::run::u64 off = 8 * (rng() % (model.size() / 8 + 2));
  if (off + 8 > kStressMaxBlob) off = 0;
  static const clio::run::u64 kClass[3] = {4096, 64 << 10, 384 << 10};
  clio::run::u64 size = 8 * (1 + rng() % (kClass[rng() % 3] / 8));
  size = std::min(size, kStressMaxBlob - off);
  const clio::run::u32 ver = ++m.vers_[j];
  const std::string b = StressBytes(id, ver, off, size);
  const clio::run::u32 rc =
      sh.io_->PutBlob(sh.tag_, StressName(id), off, size, b.data());
  sh.writes_.fetch_add(1);
  if (rc != 0) {
    sh.Report("put " + StressName(id) + " rc=" + std::to_string(rc));
    return;
  }
  if (off + size > model.size()) model.resize(off + size, '\0');
  std::memcpy(&model[off], b.data(), size);
  sh.sizes_[id - 1].store(model.size());
}

/**
 * Stress read of an owned blob: a random sub-range or the whole blob, which
 * must equal the model exactly (read-your-writes).
 */
static void StressReadOwn(StressShared &sh, std::mt19937_64 &rng,
                          StressModel &m, clio::run::u32 first) {
  const size_t j = rng() % kStressBlobsPerThread;
  const clio::run::u32 id = first + static_cast<clio::run::u32>(j);
  const std::string &model = m.bytes_[j];
  if (model.empty()) return;
  const clio::run::u64 words = model.size() / 8;
  clio::run::u64 off = 0;
  clio::run::u64 len = model.size();
  if (rng() % 2 == 0) {
    off = 8 * (rng() % words);
    len = 8 * (1 + rng() % (words - off / 8));
  }
  std::string got(len, '\0');
  const clio::run::u32 rc =
      sh.io_->GetBlob(sh.tag_, StressName(id), off, len, got.data());
  sh.reads_.fetch_add(1);
  const std::string want = model.substr(off, len);
  if (rc != 0 || got != want) {
    sh.Report("own read " + StressName(id) + " [" + std::to_string(off) +
              ", +" + std::to_string(len) + ") rc=" + std::to_string(rc) +
              ": " + StressDiff(want, got, off));
  }
}

/**
 * Stress read of a blob another thread may be writing: inside its last
 * completed size, every word must be that blob's (any write) or a hole.
 */
static void StressReadForeign(StressShared &sh, std::mt19937_64 &rng) {
  const clio::run::u32 id = 1 + static_cast<clio::run::u32>(rng() % kStressBlobs);
  const clio::run::u64 size = sh.sizes_[id - 1].load();
  if (size < 8) return;
  const clio::run::u64 words = size / 8;
  const clio::run::u64 off = 8 * (rng() % words);
  const clio::run::u64 len = 8 * (1 + rng() % (words - off / 8));
  std::string got(len, '\0');
  const clio::run::u32 rc =
      sh.io_->GetBlob(sh.tag_, StressName(id), off, len, got.data());
  sh.reads_.fetch_add(1);
  const std::string why = rc != 0 ? "rc=" + std::to_string(rc)
                                  : StressForeignCheck(got, id, off);
  if (!why.empty()) {
    sh.Report("foreign read " + StressName(id) + " [" + std::to_string(off) +
              ", +" + std::to_string(len) + "): " + why);
  }
}

/**
 * One stress thread: random writes and exact reads of its own blobs, and
 * consistency reads of everyone's, until told to stop.
 * @param sh   shared state
 * @param t    thread index (owns blobs t*kStressBlobsPerThread + 1 ...)
 * @param seed generator seed
 * @param m    out: this thread's final model
 */
static void StressThread(StressShared &sh, int t, clio::run::u64 seed,
                         StressModel *m) {
  std::mt19937_64 rng(seed + static_cast<clio::run::u64>(t) * 7919);
  m->bytes_.assign(kStressBlobsPerThread, std::string());
  m->vers_.assign(kStressBlobsPerThread, 0);
  const clio::run::u32 first =
      static_cast<clio::run::u32>(t * kStressBlobsPerThread + 1);
  while (!sh.stop_.load()) {
    const clio::run::u64 r = rng() % 10;
    if (r < 5) {
      StressWrite(sh, rng, *m, first);
    } else if (r < 8) {
      StressReadOwn(sh, rng, *m, first);
    } else {
      StressReadForeign(sh, rng);
    }
  }
}

/**
 * After the threads stop and everything is flushed, every blob must read
 * back exactly through the cache and from the chain below.
 * @param sh     shared state
 * @param models every thread's final model
 */
static void StressFinalCheck(StressShared &sh,
                             const std::vector<StressModel> &models) {
  for (int t = 0; t < kStressThreads; ++t) {
    for (int j = 0; j < kStressBlobsPerThread; ++j) {
      const std::string &want = models[t].bytes_[j];
      if (want.empty()) continue;
      const clio::run::u32 id =
          static_cast<clio::run::u32>(t * kStressBlobsPerThread + j + 1);
      std::string got(want.size(), '\0');
      const clio::run::u32 rc = sh.io_->GetBlob(sh.tag_, StressName(id), 0,
                                                got.size(), got.data());
      if (rc != 0 || got != want) {
        sh.Report("final cache read " + StressName(id) + " rc=" +
                  std::to_string(rc) + ": " + StressDiff(want, got, 0));
      }
      const std::string below = CoreRead(sh.tag_, StressName(id), 0, want.size());
      if (below != want) {
        sh.Report("final chain read " + StressName(id) + ": " +
                  StressDiff(want, below, 0));
      }
    }
  }
}

TEST_CASE("ShmWtc - stress: leased extents stay exact under concurrent puts, "
          "gets, evictions and fills",
          "[cte][shm_wtc][extents][stress]") {
  // A 2 MiB lease under a ~10 MiB working set of growing, rewritten blobs:
  // nearly every put evicts and nearly every miss fills, while other
  // threads read the same blobs and the flusher copies them out. Every
  // owner read must match its model exactly; every other read must hold
  // only the right blob's words at the right positions.
  WtcFixture fx;
  const clio::run::PoolId pool(582, 0);
  CreateCache(pool, "shm_wtc_stress", 8ull << 20, false, 2ull << 20);
  clio::cte::core::Client io(pool);
  Connect(io, pool);
  const wtc::Stats *st = io.CacheStats();
  REQUIRE(st != nullptr);
  StressShared sh;
  sh.io_ = &io;
  sh.tag_ = MakeTag(io, "wtc_stress");
  const char *ms_env = std::getenv("CLIO_WTC_STRESS_MS");
  const int ms = ms_env != nullptr ? std::atoi(ms_env) : kStressDefaultMs;
  const char *seed_env = std::getenv("CLIO_WTC_STRESS_SEED");
  const clio::run::u64 seed =
      seed_env != nullptr ? std::strtoull(seed_env, nullptr, 10)
                          : static_cast<clio::run::u64>(
                                std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count());
  INFO("stress seed " << seed << ", " << ms << " ms");
  std::vector<StressModel> models(kStressThreads);
  std::vector<std::thread> ths;
  for (int t = 0; t < kStressThreads; ++t) {
    ths.emplace_back(StressThread, std::ref(sh), t, seed, &models[t]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  sh.stop_.store(true);
  for (auto &th : ths) th.join();
  REQUIRE(WaitFlushed(io));
  StressFinalCheck(sh, models);
  INFO("stress: writes " << sh.writes_.load() << " reads " << sh.reads_.load()
       << " bad " << sh.bad_.load() << "; hits " << st->get_hits_.load()
       << " misses " << st->get_misses_.load() << " fills "
       << st->fills_.load() << "; " << RefusalCounters(st));
  for (const std::string &r : sh.reports_) INFO("  " << r);
  REQUIRE(sh.bad_.load() == 0);
  REQUIRE(st->evictions_.load() > 0);
}

SIMPLE_TEST_MAIN()
