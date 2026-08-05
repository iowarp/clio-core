/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * FULL 4-LAYER STACK TESTS (issue #886 cache/replication split)
 *
 * The canonical clio-fs chain composed at once — cache(563) ->
 * compressor(562) -> replication(561) -> core(512) — driven by ONE plain
 * clio::cte::core::Client at the top. Verifies the properties only this
 * stack has:
 *   - a compressed blob KEEPS the zero-IPC SHM fast path: the cache layer's
 *     raw REPLICA_CACHE copy serves it while the primary (stored
 *     compressed, landed with the ack — async write-through) refuses;
 *   - the replication layer's persistent replica holds the STORED form,
 *     marked transformed (the #886 per-replica transform stamping);
 *   - evicting the cache replica (capacity pass-0) drops the fast path to
 *     refusal without losing data — the task path still decompresses — and
 *     the next read re-populates the cache replica, reviving the fast path.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/cache/cache_client.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/replication/replication_client.h>

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

static const clio::run::PoolId kStackCompPoolId(562, 0);
static constexpr clio::run::u64 kValSize = 64 * 1024;

class CacheStackFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  std::string restart_log_path_;

  CacheStackFixture() {
    config_path_ = chi_test_data_dir() + "/cache_stack_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/cache_stack_file.dat";
    restart_log_path_ = chi_test_data_dir() + "/cache_stack_restart.bin";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    // Hermetic pool set: without this the runtime replays ~/.clio's restart
    // log and can resurrect a replication/cache pool from an EARLIER test's
    // compose — whose create-params then silently win over this test's
    // (GetOrCreate returns the existing pool).
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  ~CacheStackFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_);
    if (fs::exists(restart_log_path_)) fs::remove(restart_log_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    // RAM tier for primaries + cache replicas, persistent file tier for the
    // replication layer's durable copies.
    config_file << R"(
# Full 4-layer stack test configuration
runtime:
  num_threads: 2
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
      - path: "ram::cache_stack_dram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0

      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 0.2
        persistence_level: "temporary"

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

/** Highly compressible deterministic payload. */
static std::string CompressiblePayload(clio::run::u64 size) {
  std::string v(size, 'A');
  for (clio::run::u64 i = 0; i < size; i += 128) {
    v[i] = static_cast<char>('a' + (i / 128) % 26);
  }
  return v;
}

/** Poll until a replica of tag/name reports `want` bytes via the raw core. */
static bool WaitReplicaSize(clio::cte::core::Client *core,
                            const clio::cte::core::TagId &tag_id,
                            const std::string &name, clio::run::u64 want,
                            int replica) {
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto rsz = core->AsyncGetBlobSize(tag_id, name,
                                      clio::run::PoolQuery::Dynamic(),
                                      replica);
    rsz.Wait();
    if (rsz->GetReturnCode() == 0 && rsz->size_ == want) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

/** Poll until the PRIMARY exists with a nonzero size; returns it (0=timeout). */
static clio::run::u64 WaitPrimaryNonzero(clio::cte::core::Client *core,
                                         const clio::cte::core::TagId &tag_id,
                                         const std::string &name) {
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto sz = core->AsyncGetBlobSize(tag_id, name);
    sz.Wait();
    if (sz->GetReturnCode() == 0 && sz->size_ > 0) {
      return sz->size_;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return 0;
}

TEST_CASE("CacheStack - cache over compressor over replication over core",
          "[cte][cache][compressor][replicas][stack][886]") {
  CacheStackFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  // Build the chain bottom-up: replication(561) -> core(512), then
  // compressor(562) -> replication(561), then cache(563) -> compressor(562).
  {
    clio::cte::replication::Client repl(
        clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
    clio::cte::replication::ReplicationConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = repl.AsyncCreateReplication(
        clio::run::PoolQuery::Local(),
        clio::cte::replication::kReplicationPoolName,
        clio::cte::replication::kReplicationPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }
  {
    clio::cte::compressor::Client comp(kStackCompPoolId,
                                       clio::cte::core::kCtePoolId);
    clio::cte::compressor::CompressorConfig params;
    params.next_pool_id_ = clio::cte::replication::kReplicationPoolId;
    auto create = comp.AsyncCreateCompressor(
        clio::run::PoolQuery::Local(), "clio_cte_compressor_stack",
        kStackCompPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }
  {
    clio::cte::cache::Client cache(clio::cte::cache::kCachePoolId,
                                   clio::cte::core::kCtePoolId);
    clio::cte::cache::CacheConfig params;
    params.next_pool_id_ = kStackCompPoolId;
    auto create = cache.AsyncCreateCache(
        clio::run::PoolQuery::Local(), clio::cte::cache::kCachePoolName,
        clio::cte::cache::kCachePoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  // ONE plain core client at the top of the stack, mirror attached so the
  // zero-IPC fast path is live (the CLIO_CTE_POOL deployment configuration).
  clio::cte::core::Client stack_io(clio::cte::cache::kCachePoolId);
  REQUIRE(stack_io.AttachShmCacheOf(clio::cte::core::kCtePoolId));

  clio::cte::core::Tag tag("cache_stack_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string val = CompressiblePayload(kValSize);

  // ======================================================================
  // 1. A COMPRESSED put through the stack (async write-through): the
  //    compressed primary AND the raw cache replica both land before the
  //    ack, and the SHM fast path serves ORIGINAL bytes via the raw copy
  //    while the transformed primary refuses.
  // ======================================================================
  {
    clio::cte::core::Context ctx;
    ctx.compress_lib_ = 1;
    auto put = stack_io.AsyncPutBlob(tag_id, "comp_blob", 0, kValSize,
                                     val.data(), /*score=*/-1.0f, ctx);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    // Raw cache replica present NOW, at the LOGICAL size.
    {
      auto rsz = core->AsyncGetBlobSize(tag_id, "comp_blob",
                                        clio::run::PoolQuery::Dynamic(),
                                        clio::cte::core::kCacheReplica);
      rsz.Wait();
      REQUIRE(rsz->GetReturnCode() == 0);
      REQUIRE(rsz->size_ == kValSize);
    }

    // THE HEADLINE: zero-IPC fast path serves ORIGINAL bytes for a blob
    // whose authoritative form is compressed — via the raw cache replica.
    const char *view = nullptr;
    clio::run::u64 view_size = 0, gen = 0;
    REQUIRE(stack_io.TryGetBlobViewShm(tag_id, "comp_blob", &view,
                                       &view_size, &gen));
    REQUIRE(view_size == kValSize);
    REQUIRE(std::memcmp(view, val.data(), kValSize) == 0);
    REQUIRE(stack_io.CheckBlobGenShm(tag_id, "comp_blob", gen));
  }

  // ======================================================================
  // 2. The primary is COMPRESSED on the core (it landed with the ack); the
  //    persistent replica holds the STORED form and is marked TRANSFORMED
  //    (the replication sweep stamps the get's OUT flags). The fast path
  //    keeps serving raw bytes.
  // ======================================================================
  clio::run::u64 stored_size = 0;
  {
    stored_size = WaitPrimaryNonzero(core, tag_id, "comp_blob");
    REQUIRE(stored_size > 0);
    REQUIRE(stored_size < kValSize);  // really compressed on the core

    // Persistent replica (index 1) = copy of the STORED bytes.
    REQUIRE(WaitReplicaSize(core, tag_id, "comp_blob", stored_size, 1));
    {
      std::vector<char> stored(stored_size, 0), replica(stored_size, 0);
      auto g0 = core->AsyncGetBlob(tag_id, "comp_blob", 0, stored_size,
                                   /*flags=*/0, stored.data());
      g0.Wait();
      REQUIRE(g0->GetReturnCode() == 0);
      clio::cte::core::Context rctx;
      rctx.replica_ = 1;
      auto g1 = core->AsyncGetBlob(tag_id, "comp_blob", 0, stored_size,
                                   /*flags=*/0, replica.data(),
                                   clio::run::PoolQuery::Dynamic(), rctx);
      g1.Wait();
      REQUIRE(g1->GetReturnCode() == 0);
      REQUIRE(std::memcmp(stored.data(), replica.data(), stored_size) == 0);
      // Per-replica transform stamping: the get's OUT context reports the
      // persistent replica as TRANSFORMED — its bytes are codec bytes and
      // must never be served raw.
      REQUIRE(g1->context_.transform_flags_ != 0);
      // And the primary read (a DIRECT core client) returned the STORED
      // form with its flags — never short-cut onto the raw cache copy
      // (#818 stored-bytes contract; the replication sweep depends on it).
      REQUIRE(g0->context_.transform_flags_ != 0);
    }

    // The cache replica itself reports UNTRANSFORMED raw bytes.
    {
      std::vector<char> raw(kValSize, 0);
      clio::cte::core::Context cctx;
      cctx.replica_ = clio::cte::core::kCacheReplica;
      auto gc = core->AsyncGetBlob(tag_id, "comp_blob", 0, kValSize,
                                   /*flags=*/0, raw.data(),
                                   clio::run::PoolQuery::Dynamic(), cctx);
      gc.Wait();
      REQUIRE(gc->GetReturnCode() == 0);
      REQUIRE(gc->context_.transform_flags_ == 0);
      REQUIRE(std::memcmp(raw.data(), val.data(), kValSize) == 0);
    }

    // Fast path STILL serves raw bytes now that the primary is compressed.
    const char *view = nullptr;
    clio::run::u64 view_size = 0, gen = 0;
    REQUIRE(stack_io.TryGetBlobViewShm(tag_id, "comp_blob", &view,
                                       &view_size, &gen));
    REQUIRE(view_size == kValSize);
    REQUIRE(std::memcmp(view, val.data(), kValSize) == 0);

    // Task path through the stack: whole and partial reads decompress.
    std::vector<char> got(kValSize, 0);
    auto get = stack_io.AsyncGetBlob(tag_id, "comp_blob", 0, kValSize,
                                     /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);

    std::vector<char> part(777, 0);
    auto pget = stack_io.AsyncGetBlob(tag_id, "comp_blob", 5000, 777,
                                      /*flags=*/0, part.data());
    pget.Wait();
    REQUIRE(pget->GetReturnCode() == 0);
    REQUIRE(std::memcmp(part.data(), val.data() + 5000, 777) == 0);
  }

  // ======================================================================
  // 3. RECLAIM + REFILL: capacity eviction (pass-0) frees the cache
  //    replica first; the fast path drops to refusal (no raw copy left,
  //    and codec bytes must never be handed out); the task path still
  //    serves original bytes; the read re-populates the cache replica and
  //    the fast path comes back.
  // ======================================================================
  {
    // Reclaim less than the cache replica's size: pass-0 frees it and the
    // pass ends before any primary is touched.
    auto evict = core->AsyncEvict(/*min_tier_score=*/0.0f, kValSize / 2);
    evict.Wait();
    REQUIRE(evict->GetReturnCode() == 0);

    // Cache replica gone.
    {
      auto rsz = core->AsyncGetBlobSize(tag_id, "comp_blob",
                                        clio::run::PoolQuery::Dynamic(),
                                        clio::cte::core::kCacheReplica);
      rsz.Wait();
      REQUIRE((rsz->GetReturnCode() != 0 || rsz->size_ == 0));
    }

    // Fast path refuses: the primary is transformed and no raw serving
    // replica remains.
    const char *view = nullptr;
    clio::run::u64 view_size = 0, gen = 0;
    REQUIRE_FALSE(stack_io.TryGetBlobViewShm(tag_id, "comp_blob", &view,
                                             &view_size, &gen));

    // Data intact through the chain (decompressed by the compressor).
    std::vector<char> got(kValSize, 0);
    auto get = stack_io.AsyncGetBlob(tag_id, "comp_blob", 0, kValSize,
                                     /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);

    // The miss re-populated the raw cache replica; the fast path revives.
    REQUIRE(WaitReplicaSize(core, tag_id, "comp_blob", kValSize,
                            clio::cte::core::kCacheReplica));
    {
      const char *rview = nullptr;
      clio::run::u64 rview_size = 0, rgen = 0;
      bool ok = false;
      for (int attempt = 0; attempt < 100 && !ok; ++attempt) {
        ok = stack_io.TryGetBlobViewShm(tag_id, "comp_blob", &rview,
                                        &rview_size, &rgen);
        if (!ok) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      }
      REQUIRE(ok);
      REQUIRE(rview_size == kValSize);
      REQUIRE(std::memcmp(rview, val.data(), kValSize) == 0);
    }
  }
}

SIMPLE_TEST_MAIN()
