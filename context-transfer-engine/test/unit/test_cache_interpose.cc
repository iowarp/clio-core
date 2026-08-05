/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CACHE CHIMOD TESTS (issue #886 cache/replication split)
 *
 * cache(563) -> core(512) driven by ONE plain clio::cte::core::Client at the
 * top. Verifies the cache module's contract:
 *   - ASYNC WRITE-THROUGH: a put lands the AUTHORITATIVE primary below
 *     before its ack (consistency: nothing acked lives only in the cache)
 *     and updates the raw REPLICA_CACHE copy with it;
 *   - overwrites replace both copies before their ack — reads through any
 *     path observe the final bytes immediately;
 *   - the zero-IPC SHM fast path serves the blob from its (untransformed,
 *     RAM-local) copies;
 *   - a read MISS (blob written behind the cache's back) forwards down and
 *     best-effort re-populates the cache replica;
 *   - batched puts get scalar-equivalent semantics for all of the above.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/cache/cache_client.h>

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

static constexpr clio::run::u64 kValSize = 64 * 1024;

class CacheInterposeFixture {
 public:
  std::string config_path_;
  std::string restart_log_path_;

  CacheInterposeFixture() {
    config_path_ = chi_test_data_dir() + "/cache_interpose_config.yaml";
    restart_log_path_ = chi_test_data_dir() + "/cache_interpose_restart.bin";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    // Hermetic pool set: don't let ~/.clio's restart log resurrect pools
    // from earlier tests (their create-params would win over ours).
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_path_.c_str(), 1);

    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  ~CacheInterposeFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(restart_log_path_)) fs::remove(restart_log_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Cache chimod test configuration
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
      - path: "ram::cache_interpose_dram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

static std::string Payload(clio::run::u64 size, char seed) {
  std::string v(size, seed);
  for (clio::run::u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>(seed + (i % 61));
  }
  return v;
}

/** Poll until the PRIMARY of tag/name reports `want` bytes via the raw core. */
static bool WaitPrimarySize(clio::cte::core::Client *core,
                            const clio::cte::core::TagId &tag_id,
                            const std::string &name, clio::run::u64 want) {
  for (int attempt = 0; attempt < 300; ++attempt) {
    auto sz = core->AsyncGetBlobSize(tag_id, name);
    sz.Wait();
    if (sz->GetReturnCode() == 0 && sz->size_ == want) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

TEST_CASE("CacheInterpose - async write-through cache over core",
          "[cte][cache][886]") {
  CacheInterposeFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  // cache(563) -> core(512), asynchronous write-through.
  {
    clio::cte::cache::Client cache(clio::cte::cache::kCachePoolId,
                                   clio::cte::core::kCtePoolId);
    clio::cte::cache::CacheConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = cache.AsyncCreateCache(
        clio::run::PoolQuery::Local(), clio::cte::cache::kCachePoolName,
        clio::cte::cache::kCachePoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  // Task-path client at the top of the chain (no SHM mirror), plus a
  // mirror-attached one for the fast-path assertions.
  clio::cte::core::Client cache_io(clio::cte::cache::kCachePoolId);
  clio::cte::core::Client shm_io(clio::cte::cache::kCachePoolId);
  REQUIRE(shm_io.AttachShmCacheOf(clio::cte::core::kCtePoolId));

  clio::cte::core::Tag tag("cache_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string val = Payload(kValSize, 'a');

  // ======================================================================
  // 1. ASYNC WRITE-THROUGH: after the ack, BOTH copies are current — the
  //    authoritative primary below (consistency: nothing acked lives only
  //    in the cache) AND the raw cache replica the fast path serves from.
  // ======================================================================
  {
    auto put = cache_io.AsyncPutBlob(tag_id, "wb_blob", 0, kValSize,
                                     val.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    // The authoritative primary landed BEFORE the ack — no window where
    // the only copy of acked data is the cache.
    {
      auto psz = core->AsyncGetBlobSize(tag_id, "wb_blob");
      psz.Wait();
      REQUIRE(psz->GetReturnCode() == 0);
      REQUIRE(psz->size_ == kValSize);
    }

    // Single node ⇒ this node OWNS every blob, and the owner keeps NO
    // cache-slot copy (issue #894): the primary IS the node-local copy,
    // and an owner-side copy would sit outside the invalidation protocol.
    {
      auto rsz = core->AsyncGetBlobSize(tag_id, "wb_blob",
                                        clio::run::PoolQuery::Dynamic(),
                                        clio::cte::core::kCacheReplica);
      rsz.Wait();
      REQUIRE((rsz->GetReturnCode() != 0 || rsz->size_ == 0));
    }

    // Size through the cache answers the logical size from the raw copy.
    {
      auto sz = cache_io.AsyncGetBlobSize(tag_id, "wb_blob");
      sz.Wait();
      REQUIRE(sz->GetReturnCode() == 0);
      REQUIRE(sz->size_ == kValSize);
    }

    // Read through the cache: served from the cache replica.
    {
      std::vector<char> got(kValSize, 0);
      auto get = cache_io.AsyncGetBlob(tag_id, "wb_blob", 0, kValSize,
                                       /*flags=*/0, got.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
    }

    // Zero-IPC fast path serves the blob (primary or serving replica —
    // both are current and untransformed here).
    {
      const char *view = nullptr;
      clio::run::u64 view_size = 0, gen = 0;
      REQUIRE(shm_io.TryGetBlobViewShm(tag_id, "wb_blob", &view, &view_size,
                                       &gen));
      REQUIRE(view_size == kValSize);
      REQUIRE(std::memcmp(view, val.data(), kValSize) == 0);
      REQUIRE(shm_io.CheckBlobGenShm(tag_id, "wb_blob", gen));
    }

    // And the primary holds the same bytes.
    {
      std::vector<char> got(kValSize, 0);
      auto get = core->AsyncGetBlob(tag_id, "wb_blob", 0, kValSize,
                                    /*flags=*/0, got.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
    }
  }

  // ======================================================================
  // 2. OVERWRITE: the second put replaces BOTH copies before its ack —
  //    reads through any path observe the final bytes immediately.
  // ======================================================================
  {
    const std::string v1 = Payload(kValSize, 'b');
    const std::string v2 = Payload(kValSize, 'c');
    auto p1 = cache_io.AsyncPutBlob(tag_id, "co_blob", 0, kValSize,
                                    v1.data());
    p1.Wait();
    REQUIRE(p1->GetReturnCode() == 0);
    auto p2 = cache_io.AsyncPutBlob(tag_id, "co_blob", 0, kValSize,
                                    v2.data());
    p2.Wait();
    REQUIRE(p2->GetReturnCode() == 0);

    // The primary holds the second payload, no waiting. (No cache-slot
    // copy to check — the owner node keeps none, issue #894.)
    std::vector<char> got(kValSize, 0);
    auto get = core->AsyncGetBlob(tag_id, "co_blob", 0, kValSize,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v2.data(), kValSize) == 0);
    // Reads through the cache observe the final bytes too.
    {
      std::vector<char> via(kValSize, 0);
      auto gc = cache_io.AsyncGetBlob(tag_id, "co_blob", 0, kValSize,
                                      /*flags=*/0, via.data());
      gc.Wait();
      REQUIRE(gc->GetReturnCode() == 0);
      REQUIRE(std::memcmp(via.data(), v2.data(), kValSize) == 0);
    }
  }

  // ======================================================================
  // 3. MISS + REPOPULATE: a blob written straight to the core (behind the
  //    cache's back) reads correctly through the cache, and the read
  //    re-populates the cache replica.
  // ======================================================================
  {
    const std::string vm = Payload(kValSize, 'd');
    auto put = core->AsyncPutBlob(tag_id, "miss_blob", 0, kValSize,
                                  vm.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    std::vector<char> got(kValSize, 0);
    auto get = cache_io.AsyncGetBlob(tag_id, "miss_blob", 0, kValSize,
                                     /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), vm.data(), kValSize) == 0);

    // The owner node never populates a cache-slot copy (issue #894): the
    // read was served straight from the node-local primary.
    {
      auto rsz = core->AsyncGetBlobSize(tag_id, "miss_blob",
                                        clio::run::PoolQuery::Dynamic(),
                                        clio::cte::core::kCacheReplica);
      rsz.Wait();
      REQUIRE((rsz->GetReturnCode() != 0 || rsz->size_ == 0));
    }
  }

  // ======================================================================
  // 4. BATCHED puts get SCALAR-EQUIVALENT semantics (issue #886 follow-up):
  //    one MultiPutBlob through the cache lands EVERY record's primary
  //    below AND its raw cache replica before the ack, and the SHM fast
  //    path serves each record.
  // ======================================================================
  {
    auto *ipc_manager = CLIO_CPU_IPC;
    constexpr size_t kRecords = 8;
    constexpr clio::run::u64 kRecSize = 4096;
    std::vector<std::string> vals;
    ctp::ipc::FullPtr<char> staging =
        ipc_manager->AllocateBuffer(kRecords * kRecSize);
    REQUIRE(!staging.IsNull());
    std::vector<clio::cte::core::MultiPutDesc> descs(kRecords);
    for (size_t i = 0; i < kRecords; ++i) {
      vals.push_back(Payload(kRecSize, static_cast<char>('e' + i)));
      std::memcpy(staging.ptr_ + i * kRecSize, vals[i].data(), kRecSize);
      descs[i].tag_id_ = tag_id;
      descs[i].blob_name_ = "batch_blob_" + std::to_string(i);
      descs[i].size_ = kRecSize;
      descs[i].payload_off_ = i * kRecSize;
    }
    auto fut = cache_io.AsyncMultiPutVectored(
        ctp::ipc::ShmPtr<>(staging.shm_), kRecords * kRecSize, descs);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);

    for (size_t i = 0; i < kRecords; ++i) {
      const std::string name = "batch_blob_" + std::to_string(i);
      // Authoritative primary landed with the ack.
      {
        auto psz = core->AsyncGetBlobSize(tag_id, name);
        psz.Wait();
        REQUIRE(psz->GetReturnCode() == 0);
        REQUIRE(psz->size_ == kRecSize);
      }
      // Raw cache replica exists NOW too.
      auto rsz = core->AsyncGetBlobSize(tag_id, name,
                                        clio::run::PoolQuery::Dynamic(),
                                        clio::cte::core::kCacheReplica);
      rsz.Wait();
      REQUIRE(rsz->GetReturnCode() == 0);
      REQUIRE(rsz->size_ == kRecSize);
      // Zero-IPC fast path serves the record.
      const char *view = nullptr;
      clio::run::u64 view_size = 0, gen = 0;
      REQUIRE(shm_io.TryGetBlobViewShm(tag_id, name, &view, &view_size,
                                       &gen));
      REQUIRE(view_size == kRecSize);
      REQUIRE(std::memcmp(view, vals[i].data(), kRecSize) == 0);
      // And the primary's bytes match.
      std::vector<char> got(kRecSize, 0);
      auto g = core->AsyncGetBlob(tag_id, name, 0, kRecSize, /*flags=*/0,
                                  got.data());
      g.Wait();
      REQUIRE(g->GetReturnCode() == 0);
      REQUIRE(std::memcmp(got.data(), vals[i].data(), kRecSize) == 0);
    }
  }
}

SIMPLE_TEST_MAIN()
