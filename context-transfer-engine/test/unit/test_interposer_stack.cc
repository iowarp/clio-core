/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * FULL INTERPOSER STACK TESTS (issue #886)
 *
 * The whole chain composed at once — compressor(562) → replication(561) →
 * core(512) — driven by ONE plain clio::cte::core::Client at the top, with
 * the client-side SHM mirror attached. Verifies the three things only the
 * full stack can:
 *   - the layers compose: a compressed put lands compressed on the core AND
 *     gains an (async) persistent replica of the STORED bytes; a dropped
 *     primary heals through replication and still decompresses on the way
 *     out of the compressor;
 *   - the zero-IPC SHM fast path stays FUNCTIONAL through the stack for
 *     untransformed blobs (the mirror serves them without any task);
 *   - the fast path REFUSES transformed blobs (#818) — a mirror-attached
 *     client must never be handed codec bytes that the task path (through
 *     the compressor) would have decompressed.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
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

class InterposerStackFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;

  InterposerStackFixture() {
    config_path_ = chi_test_data_dir() + "/interposer_stack_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/interposer_stack_file.dat";
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

  ~InterposerStackFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    // RAM tier for primaries + a persistent file tier for the replication
    // layer's durable copies.
    config_file << R"(
# Full interposer stack test configuration
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
      - path: "ram::interposer_stack_dram"
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

/** Poll until replica 1 of tag/name reports `want` bytes via the raw core. */
static bool WaitReplicaSize(clio::cte::core::Client *core,
                            const clio::cte::core::TagId &tag_id,
                            const std::string &name, clio::run::u64 want) {
  for (int attempt = 0; attempt < 250; ++attempt) {
    auto rsz = core->AsyncGetBlobSize(tag_id, name,
                                      clio::run::PoolQuery::Dynamic(), 1);
    rsz.Wait();
    if (rsz->GetReturnCode() == 0 && rsz->size_ == want) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

TEST_CASE("InterposerStack - compressor over replication over core",
          "[cte][compressor][replicas][stack][886]") {
  InterposerStackFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  // Build the chain bottom-up: replication(561) -> core(512), then
  // compressor(562) -> replication(561).
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

  // ONE plain core client at the top of the stack, mirror attached so the
  // zero-IPC fast path is live (the CLIO_CTE_POOL deployment configuration).
  clio::cte::core::Client stack_io(kStackCompPoolId);
  REQUIRE(stack_io.AttachShmCacheOf(clio::cte::core::kCtePoolId));

  clio::cte::core::Tag tag("stack_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string val = CompressiblePayload(kValSize);

  // ======================================================================
  // 1. UNCOMPRESSED blob through the stack: readable, replicated, and the
  //    SHM fast path serves it with zero IPC.
  // ======================================================================
  {
    auto put = stack_io.AsyncPutBlob(tag_id, "plain_blob", 0, kValSize,
                                     val.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    // The replication layer picked the put up (async sweep).
    REQUIRE(WaitReplicaSize(core, tag_id, "plain_blob", kValSize));

    // Task-path read through the whole stack.
    std::vector<char> got(kValSize, 0);
    auto get = stack_io.AsyncGetBlob(tag_id, "plain_blob", 0, kValSize,
                                     /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);

    // Zero-IPC fast path FUNCTIONAL: the mirror hands out a direct view of
    // the untransformed primary.
    const char *view = nullptr;
    clio::run::u64 view_size = 0, gen = 0;
    REQUIRE(stack_io.TryGetBlobViewShm(tag_id, "plain_blob", &view,
                                       &view_size, &gen));
    REQUIRE(view_size == kValSize);
    REQUIRE(std::memcmp(view, val.data(), kValSize) == 0);
    REQUIRE(stack_io.CheckBlobGenShm(tag_id, "plain_blob", gen));
  }

  // ======================================================================
  // 2. COMPRESSED blob through the stack: stored compressed on the core,
  //    replica holds the STORED bytes, the fast path REFUSES it, and the
  //    task path decompresses.
  // ======================================================================
  clio::run::u64 stored_size = 0;
  {
    clio::cte::core::Context ctx;
    ctx.compress_lib_ = 1;
    auto put = stack_io.AsyncPutBlob(tag_id, "comp_blob", 0, kValSize,
                                     val.data(), /*score=*/-1.0f, ctx);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    // Compressed on the core: stored size strictly smaller than logical.
    auto raw_sz = core->AsyncGetBlobSize(tag_id, "comp_blob");
    raw_sz.Wait();
    REQUIRE(raw_sz->GetReturnCode() == 0);
    stored_size = raw_sz->size_;
    REQUIRE(stored_size > 0);
    REQUIRE(stored_size < kValSize);

    // The replica is a durable copy of the STORED (compressed) form.
    REQUIRE(WaitReplicaSize(core, tag_id, "comp_blob", stored_size));
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
    }

    // THE #818 GUARANTEE THROUGH THE STACK: the mirror-attached fast path
    // must refuse the transformed blob — codec bytes must never short-cut
    // past the compressor.
    const char *view = nullptr;
    clio::run::u64 view_size = 0, gen = 0;
    REQUIRE_FALSE(stack_io.TryGetBlobViewShm(tag_id, "comp_blob", &view,
                                             &view_size, &gen));

    // And the task path (which the refusal forces) returns ORIGINAL bytes —
    // full and partial (vectored covered by cte_compressor_interpose).
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
  // 3. Drop + heal through the FULL stack: the organizer drops the primary
  //    (persistent replica exists), a read through the stack serves from
  //    the replica via the replication layer, re-caches the primary, and
  //    the compressor still hands back ORIGINAL bytes.
  // ======================================================================
  {
    auto reorg = core->AsyncReorganizeBlob(tag_id, "comp_blob", 0.05f);
    reorg.Wait();
    REQUIRE(reorg->GetReturnCode() == 0);
    {
      auto psz = core->AsyncGetBlobSize(tag_id, "comp_blob");
      psz.Wait();
      REQUIRE(psz->size_ == 0);  // primary really dropped
    }

    std::vector<char> got(kValSize, 0);
    auto get = stack_io.AsyncGetBlob(tag_id, "comp_blob", 0, kValSize,
                                     /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);

    // Healed: the primary is back at its stored (compressed) size.
    {
      auto psz = core->AsyncGetBlobSize(tag_id, "comp_blob");
      psz.Wait();
      REQUIRE(psz->GetReturnCode() == 0);
      REQUIRE(psz->size_ == stored_size);
    }

    // Still transformed, still refused by the fast path after the heal.
    const char *view = nullptr;
    clio::run::u64 view_size = 0, gen = 0;
    REQUIRE_FALSE(stack_io.TryGetBlobViewShm(tag_id, "comp_blob", &view,
                                             &view_size, &gen));
  }
}

SIMPLE_TEST_MAIN()
