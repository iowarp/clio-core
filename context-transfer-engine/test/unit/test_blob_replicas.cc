/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * BLOB REPLICA TESTS (issue #886)
 *
 * The CTE mechanism:
 *  - Context::replica_ == N > 0 targets replica N on put/get; a first write
 *    creates the replica lazily; primary bytes are untouched by replica
 *    writes and vice versa (isolation).
 *  - Reads of a replica no write created fail cleanly, as do reads with
 *    replica_ == kAllReplicas (a write-through selector, not a source).
 *  - Context::replica_ == kAllReplicas writes through: primary AND every
 *    existing replica receive the bytes under one write-token hold.
 *  - DelBlob destroys replicas with the blob (their blocks are freed).
 *
 * The replication chimod's policy verbs on top:
 *  - ReplicateBlob copies one blob's primary into a chosen replica.
 *  - FlushTag sweeps every qualifying blob in a tag into a replica.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
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

static constexpr clio::run::u64 kValSize = 4096;

class BlobReplicasFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  std::string metadata_log_path_;

  BlobReplicasFixture() {
    config_path_ = chi_test_data_dir() + "/blob_replicas_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/blob_replicas_file.dat";
    metadata_log_path_ = chi_test_data_dir() + "/blob_replicas_meta.log";
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

  ~BlobReplicasFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_);
    // The WAL is a file family: the snapshot plus per-shard .blob.N logs.
    std::error_code ec;
    for (const auto &entry :
         fs::directory_iterator(fs::path(metadata_log_path_).parent_path(),
                                ec)) {
      const std::string name = entry.path().filename().string();
      if (name.rfind("blob_replicas_meta.log", 0) == 0) {
        fs::remove(entry.path(), ec);
      }
    }
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    // Two tiers: a fast VOLATILE RAM tier and a small non-volatile file
    // tier. The file tier is deliberately tiny (1MB) so the
    // REPLICA_PERSISTENT test can prove enforcement by exhaustion: a
    // persistent replica larger than 1MB has nowhere legal to go and must
    // fail, while the same payload without the flag lands in RAM.
    config_file << R"(
# Blob replica test configuration - RAM (volatile) + small file (temporary)
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
      - path: "ram::blob_replicas_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0

      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "1MB"
        score: 0.2
        persistence_level: "temporary"

    dpe:
      dpe_type: "max_bw"

    # WAL on (issue #886): replica writes append kExtendReplica records and
    # the explicit FlushMetadata below snapshots replica layouts (entry type
    # 3). Periodic flushing off so the only snapshot is the test's own.
    performance:
      metadata_log_path: )" << metadata_log_path_ << R"(
      transaction_log_capacity: 4MB
      flush_metadata_period_ms: 0
      flush_data_period_ms: 0
)";
    config_file.close();
  }
};

static clio::cte::core::Context ReplicaCtx(int replica) {
  clio::cte::core::Context ctx;
  ctx.replica_ = replica;
  return ctx;
}

/** Put `val` into the given replica (0 = primary) of tag/name. */
static void PutTo(clio::cte::core::Client *client,
                  const clio::cte::core::TagId &tag_id,
                  const std::string &name, const std::string &val,
                  int replica) {
  auto fut = client->AsyncPutBlob(tag_id, name, 0, val.size(), val.data(),
                                  /*score=*/-1.0f, ReplicaCtx(replica));
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
}

/** Read kValSize bytes from the given replica; returns the task rc. */
static clio::run::u32 GetFrom(clio::cte::core::Client *client,
                              const clio::cte::core::TagId &tag_id,
                              const std::string &name, std::vector<char> *out,
                              int replica) {
  out->assign(kValSize, 0);
  auto fut = client->AsyncGetBlob(tag_id, name, 0, kValSize, /*flags=*/0,
                                  out->data(), clio::run::PoolQuery::Dynamic(),
                                  ReplicaCtx(replica));
  fut.Wait();
  return fut->GetReturnCode();
}

TEST_CASE("BlobReplicas - replica write/read isolation",
          "[cte][replicas][886]") {
  BlobReplicasFixture fixture;
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_iso_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 'P');
  const std::string replica_val(kValSize, 'R');

  // Primary put, then a DIFFERENT payload into replica 1.
  PutTo(client, tag_id, "iso_blob", primary_val, 0);
  PutTo(client, tag_id, "iso_blob", replica_val, 1);

  // Primary read is untouched by the replica write; replica read returns the
  // replica's bytes.
  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), primary_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);

  // A replica no write created fails cleanly, as does the write-through
  // selector used as a read source.
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 2) != 0);
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got,
                  clio::cte::core::kAllReplicas) != 0);

  // A later primary overwrite leaves the replica's bytes alone.
  const std::string primary_v2(kValSize, 'Q');
  PutTo(client, tag_id, "iso_blob", primary_v2, 0);
  REQUIRE(GetFrom(client, tag_id, "iso_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
}

TEST_CASE("BlobReplicas - write-through to all replicas",
          "[cte][replicas][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_wt_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Seed primary + replica 1 with distinct old bytes.
  PutTo(client, tag_id, "wt_blob", std::string(kValSize, 'p'), 0);
  PutTo(client, tag_id, "wt_blob", std::string(kValSize, 'r'), 1);

  // Write-through: one put updates primary AND the existing replica.
  const std::string new_val(kValSize, 'W');
  PutTo(client, tag_id, "wt_blob", new_val, clio::cte::core::kAllReplicas);

  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "wt_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), new_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "wt_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), new_val.data(), kValSize) == 0);

  // Write-through does NOT invent replicas: replica 2 still doesn't exist.
  REQUIRE(GetFrom(client, tag_id, "wt_blob", &got, 2) != 0);
}

TEST_CASE("BlobReplicas - delete destroys replicas with the blob",
          "[cte][replicas][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_del_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  PutTo(client, tag_id, "del_blob", std::string(kValSize, 'p'), 0);
  PutTo(client, tag_id, "del_blob", std::string(kValSize, 'r'), 1);

  auto del = client->AsyncDelBlob(tag_id, "del_blob");
  del.Wait();
  REQUIRE(del->GetReturnCode() == 0);

  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "del_blob", &got, 0) != 0);
  REQUIRE(GetFrom(client, tag_id, "del_blob", &got, 1) != 0);
}

TEST_CASE("BlobReplicas - per-replica score drives reorganizer migration",
          "[cte][replicas][reorganize][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_reorg_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 'p');
  const std::string replica_val(kValSize, 'r');
  PutTo(client, tag_id, "reorg_blob", primary_val, 0);
  PutTo(client, tag_id, "reorg_blob", replica_val, 1);

  // Migrate the REPLICA to the slow tier by its own score; the primary's
  // score and placement are untouched.
  auto move_down = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 0.2f, clio::run::PoolQuery::Dynamic(), /*replica=*/1);
  move_down.Wait();
  REQUIRE(move_down->GetReturnCode() == 0);

  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "reorg_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "reorg_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), primary_val.data(), kValSize) == 0);

  // The replica remembers ITS score: the same target score again is a
  // below-threshold no-op, and moving back up works.
  auto again = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 0.2f, clio::run::PoolQuery::Dynamic(), 1);
  again.Wait();
  REQUIRE(again->GetReturnCode() == 0);
  auto move_up = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 1.0f, clio::run::PoolQuery::Dynamic(), 1);
  move_up.Wait();
  REQUIRE(move_up->GetReturnCode() == 0);
  REQUIRE(GetFrom(client, tag_id, "reorg_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);

  // Reorganizing a replica no write created fails cleanly.
  auto absent = client->AsyncReorganizeBlob(
      tag_id, "reorg_blob", 0.5f, clio::run::PoolQuery::Dynamic(), 7);
  absent.Wait();
  REQUIRE(absent->GetReturnCode() != 0);
}

TEST_CASE("BlobReplicas - REPLICA_FIXED pins a replica against the organizer",
          "[cte][replicas][reorganize][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_fixed_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string replica_val(kValSize, 'F');
  PutTo(client, tag_id, "fixed_blob", std::string(kValSize, 'p'), 0);
  {
    clio::cte::core::Context ctx = ReplicaCtx(1);
    ctx.replica_flags_ = clio::cte::core::REPLICA_FIXED;
    auto fut = client->AsyncPutBlob(tag_id, "fixed_blob", 0, kValSize,
                                    replica_val.data(), /*score=*/-1.0f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }

  // The reorganizer must not touch a FIXED replica: success (the organizer
  // sweeps blindly; FIXED is the filter), bytes stay put.
  auto fut = client->AsyncReorganizeBlob(
      tag_id, "fixed_blob", 0.2f, clio::run::PoolQuery::Dynamic(), 1);
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "fixed_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
}

TEST_CASE("BlobReplicas - REPLICA_PERSISTENT excludes volatile tiers",
          "[cte][replicas][persistence][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_persist_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // A payload bigger than the ONLY non-volatile tier (1MB file): as a plain
  // replica it lands in RAM; as a PERSISTENT replica it has nowhere legal
  // to go and the put must fail — proof the flag really excludes the
  // volatile tier rather than merely preferring the file one.
  const clio::run::u64 kBig = 4ULL * 1024 * 1024;
  const std::string big(kBig, 'B');
  {
    auto fut = client->AsyncPutBlob(tag_id, "persist_blob", 0, kBig,
                                    big.data(), -1.0f, ReplicaCtx(1));
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
  {
    clio::cte::core::Context ctx = ReplicaCtx(2);
    ctx.replica_flags_ = clio::cte::core::REPLICA_PERSISTENT;
    auto fut = client->AsyncPutBlob(tag_id, "persist_blob", 0, kBig,
                                    big.data(), -1.0f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() != 0);
  }

  // A small PERSISTENT replica fits the file tier; seed it with a LOW score,
  // then reorganize it toward the volatile tier's score — the move must
  // succeed while staying on non-volatile storage (rc 0, bytes intact).
  const clio::run::u64 kSmall = 64 * 1024;
  const std::string small_val(kSmall, 'S');
  PutTo(client, tag_id, "persist_small", std::string(kSmall, 'p'), 0);
  {
    clio::cte::core::Context ctx = ReplicaCtx(1);
    ctx.replica_flags_ = clio::cte::core::REPLICA_PERSISTENT;
    auto fut = client->AsyncPutBlob(tag_id, "persist_small", 0, kSmall,
                                    small_val.data(), /*score=*/0.2f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
  {
    auto fut = client->AsyncReorganizeBlob(
        tag_id, "persist_small", 1.0f, clio::run::PoolQuery::Dynamic(), 1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    std::vector<char> got(kSmall, 0);
    auto get = client->AsyncGetBlob(tag_id, "persist_small", 0, kSmall,
                                    /*flags=*/0, got.data(),
                                    clio::run::PoolQuery::Dynamic(),
                                    ReplicaCtx(1));
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), small_val.data(), kSmall) == 0);
  }
}

TEST_CASE("BlobReplicas - replication module ReplicateBlob and FlushTag",
          "[cte][replicas][replication][886]") {
  auto *core_client = CLIO_CTE_CLIENT;
  REQUIRE(core_client != nullptr);

  // Create/bind the replication pool over the default CTE core pool.
  clio::cte::replication::Client repl(
      clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
  {
    clio::cte::replication::ReplicationConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = repl.AsyncCreateReplication(
        clio::run::PoolQuery::Local(),
        clio::cte::replication::kReplicationPoolName,
        clio::cte::replication::kReplicationPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  clio::cte::core::Tag tag("replica_mod_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Seed primaries only.
  constexpr int kNumBlobs = 5;
  for (int i = 0; i < kNumBlobs; ++i) {
    PutTo(core_client, tag_id, "mod_blob_" + std::to_string(i),
          std::string(kValSize, static_cast<char>('a' + i)), 0);
  }

  // ReplicateBlob: one blob into replica 1; the replica must read back the
  // primary's bytes.
  {
    auto fut = repl.AsyncReplicateBlob(tag_id, "mod_blob_0", 1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->bytes_copied_ == kValSize);
    std::vector<char> got;
    REQUIRE(GetFrom(core_client, tag_id, "mod_blob_0", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), std::string(kValSize, 'a').data(),
                        kValSize) == 0);
  }

  // FlushTag: every blob in the tag gains an up-to-date replica 1.
  {
    auto fut = repl.AsyncFlushTag(tag_id, /*replica=*/1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->blobs_replicated_ == kNumBlobs);
    for (int i = 0; i < kNumBlobs; ++i) {
      std::vector<char> got;
      REQUIRE(GetFrom(core_client, tag_id, "mod_blob_" + std::to_string(i),
                      &got, 1) == 0);
      REQUIRE(std::memcmp(got.data(),
                          std::string(kValSize, static_cast<char>('a' + i))
                              .data(),
                          kValSize) == 0);
    }
  }

  // Replica stays a snapshot until re-replicated: change a primary, replica
  // still holds old bytes; ReplicateBlob refreshes it.
  {
    const std::string v2(kValSize, 'Z');
    PutTo(core_client, tag_id, "mod_blob_1", v2, 0);
    std::vector<char> got;
    REQUIRE(GetFrom(core_client, tag_id, "mod_blob_1", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), std::string(kValSize, 'b').data(),
                        kValSize) == 0);
    auto fut = repl.AsyncReplicateBlob(tag_id, "mod_blob_1", 1);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(GetFrom(core_client, tag_id, "mod_blob_1", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), v2.data(), kValSize) == 0);
  }
}

TEST_CASE("BlobReplicas - reorganizer drops the cache copy below a "
          "persistent replica",
          "[cte][replicas][cache][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_drop_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 'p');
  const std::string replica_val(kValSize, 'q');
  PutTo(client, tag_id, "drop_blob", primary_val, 0);
  {
    // Persistent copy at a modest score — the organizer's drop threshold.
    clio::cte::core::Context ctx = ReplicaCtx(1);
    ctx.replica_flags_ = clio::cte::core::REPLICA_PERSISTENT;
    auto fut = client->AsyncPutBlob(tag_id, "drop_blob", 0, kValSize,
                                    replica_val.data(), /*score=*/0.5f, ctx);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }

  // Rescoring the primary BELOW the persistent replica must DROP it (free
  // its blocks) rather than migrate it — a durable copy already exists.
  {
    auto fut = client->AsyncReorganizeBlob(tag_id, "drop_blob", 0.1f);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
  {
    auto sz = client->AsyncGetBlobSize(tag_id, "drop_blob");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    REQUIRE(sz->size_ == 0);  // primary dropped
    auto rsz = client->AsyncGetBlobSize(tag_id, "drop_blob",
                                        clio::run::PoolQuery::Dynamic(), 1);
    rsz.Wait();
    REQUIRE(rsz->GetReturnCode() == 0);
    REQUIRE(rsz->size_ == kValSize);  // replica untouched
    std::vector<char> got;
    REQUIRE(GetFrom(client, tag_id, "drop_blob", &got, 1) == 0);
    REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
  }

  // Re-scoring the (now empty) primary is a clean no-op.
  {
    auto fut = client->AsyncReorganizeBlob(tag_id, "drop_blob", 0.9f);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
  }
}

TEST_CASE("BlobReplicas - transparent interposition on the core Put/Get",
          "[cte][replicas][replication][cache][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  // Bind the replication pool (GetOrCreatePool is idempotent; the module
  // test earlier created it with default params: num_replicas=1,
  // cache_score=1.0, replica_score=0.2).
  clio::cte::replication::Client repl(
      clio::cte::replication::kReplicationPoolId, clio::cte::core::kCtePoolId);
  {
    clio::cte::replication::ReplicationConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = repl.AsyncCreateReplication(
        clio::run::PoolQuery::Local(),
        clio::cte::replication::kReplicationPoolName,
        clio::cte::replication::kReplicationPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  // The whole point (issue #886): a PLAIN core client pointed at the
  // replication pool — no new client type, no new verbs.
  clio::cte::core::Client repl_io(clio::cte::replication::kReplicationPoolId);

  clio::cte::core::Tag tag("cache_model_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string val(kValSize, 'C');

  // 1. Write-through: one DEFAULT put through the interposer updates the
  //    DRAM primary NOW and the fixed persistent set ASYNCHRONOUSLY (the
  //    put acks after the primary; the periodic sweep replicates within
  //    replicate_period_ms). Primary is checked immediately, the replica by
  //    bounded poll.
  PutTo(&repl_io, tag_id, "cache_blob", val, 0);
  {
    auto psz = client->AsyncGetBlobSize(tag_id, "cache_blob");
    psz.Wait();
    REQUIRE(psz->size_ == kValSize);
  }
  {
    bool replicated = false;
    for (int attempt = 0; attempt < 200 && !replicated; ++attempt) {
      auto rsz = client->AsyncGetBlobSize(tag_id, "cache_blob",
                                          clio::run::PoolQuery::Dynamic(), 1);
      rsz.Wait();
      replicated = rsz->GetReturnCode() == 0 && rsz->size_ == kValSize;
      if (!replicated) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }
    REQUIRE(replicated);  // replica created by the async sweep
  }

  // 2. Cache hit: served from the primary through the interposer.
  {
    std::vector<char> got;
    REQUIRE(GetFrom(&repl_io, tag_id, "cache_blob", &got, 0) == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
  }

  // 3. The organizer drops the cache copy (primary score sinks below the
  //    persistent replica's replica_score). Raw core sizes show the drop;
  //    the INTERPOSED size stays logical (the replica still has the bytes),
  //    so size-then-read callers never see the blob vanish.
  {
    auto fut = client->AsyncReorganizeBlob(tag_id, "cache_blob", 0.05f);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    auto psz = client->AsyncGetBlobSize(tag_id, "cache_blob");
    psz.Wait();
    REQUIRE(psz->size_ == 0);  // raw primary really dropped
    auto lsz = repl_io.AsyncGetBlobSize(tag_id, "cache_blob");
    lsz.Wait();
    REQUIRE(lsz->GetReturnCode() == 0);
    REQUIRE(lsz->size_ == kValSize);  // logical size via the interposer
  }

  // 4. Cache miss: the interposed get serves from the persistent replica
  //    and re-populates the DRAM primary in full.
  {
    std::vector<char> got;
    REQUIRE(GetFrom(&repl_io, tag_id, "cache_blob", &got, 0) == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
    auto psz = client->AsyncGetBlobSize(tag_id, "cache_blob");
    psz.Wait();
    REQUIRE(psz->size_ == kValSize);  // primary restored by the re-cache
  }

  // 5. Fast path restored: the next read is a primary hit again.
  {
    std::vector<char> got;
    REQUIRE(GetFrom(&repl_io, tag_id, "cache_blob", &got, 0) == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
  }

  // 6. Metadata ops forward untouched: a tag op through the interposer
  //    behaves exactly like the core pool's.
  {
    auto sz = repl_io.AsyncGetTagSize(tag_id);
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
  }
}

TEST_CASE("BlobReplicas - RegisterReplicaContainer coherence smoke",
          "[cte][replicas][coherence][886]") {
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_coherence_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  // Registering against a blob that does not exist is rejected — there is
  // nothing to be coherent with.
  {
    auto reg = client->AsyncRegisterReplicaContainer(tag_id, "no_such_blob",
                                                     /*node_id=*/12345);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() != 0);
  }

  const std::string v1(kValSize, '1');
  PutTo(client, tag_id, "coherent_blob", v1, 0);

  // The owner's OWN node registering is accepted as a no-op: a registered
  // self would make the next put's invalidation delete the authoritative
  // blob. The subsequent put must therefore leave the blob fully intact.
  {
    auto reg = client->AsyncRegisterReplicaContainer(
        tag_id, "coherent_blob", CLIO_CPU_IPC->GetNodeId());
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
  }
  const std::string v2(kValSize, '2');
  PutTo(client, tag_id, "coherent_blob", v2, 0);
  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "coherent_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), v2.data(), kValSize) == 0);
}

TEST_CASE("BlobReplicas - metadata snapshot covers replica layouts",
          "[cte][replicas][wal][886]") {
  // The fixture's config enables the WAL, so every replica write in this
  // suite has been appending kExtendReplica records. This case drives the
  // OTHER half of durability: FlushMetadata's snapshot, which must emit the
  // per-replica entries (type 3) alongside the tag/blob entries — the WAL is
  // truncated after a snapshot, so a snapshot that skipped replicas would
  // lose every replica layout on restart.
  auto *client = CLIO_CTE_CLIENT;
  REQUIRE(client != nullptr);

  clio::cte::core::Tag tag("replica_snapshot_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  const std::string primary_val(kValSize, 's');
  const std::string replica_val(kValSize, 't');
  PutTo(client, tag_id, "snap_blob", primary_val, 0);
  PutTo(client, tag_id, "snap_blob", replica_val, 1);

  auto fut = client->AsyncFlushMetadata(clio::run::PoolQuery::Local(), 0);
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
  // At minimum: this case's tag entry, its blob entry, and its replica entry.
  REQUIRE(fut->entries_flushed_ >= 3);

  // The snapshot is a read-only pass: both copies still serve their bytes.
  std::vector<char> got;
  REQUIRE(GetFrom(client, tag_id, "snap_blob", &got, 0) == 0);
  REQUIRE(std::memcmp(got.data(), primary_val.data(), kValSize) == 0);
  REQUIRE(GetFrom(client, tag_id, "snap_blob", &got, 1) == 0);
  REQUIRE(std::memcmp(got.data(), replica_val.data(), kValSize) == 0);
}

TEST_CASE("BlobReplicas - ReplicationConfig parses compose YAML",
          "[cte][replicas][replication][886]") {
  // The chimod's LoadConfig is what turns a clio_cte_replication compose
  // entry (clio_default.yaml ships one) into runtime parameters.
  using clio::cte::replication::ReplicationConfig;
  {
    clio::run::PoolConfig pool_config;
    pool_config.config_ =
        "next_pool_id: \"512.7\"\n"
        "num_replicas: 3\n"
        "cache_score: 0.9\n"
        "replica_score: 0.35\n";
    ReplicationConfig config;
    config.LoadConfig(pool_config);
    REQUIRE(config.next_pool_id_ == clio::run::PoolId(512, 7));
    REQUIRE(config.num_replicas_ == 3);
    REQUIRE(config.cache_score_ == 0.9f);
    REQUIRE(config.replica_score_ == 0.35f);
  }
  // Empty config: every default survives.
  {
    clio::run::PoolConfig pool_config;
    ReplicationConfig config;
    ReplicationConfig defaults;
    config.LoadConfig(pool_config);
    REQUIRE(config.num_replicas_ == defaults.num_replicas_);
    REQUIRE(config.cache_score_ == defaults.cache_score_);
    REQUIRE(config.replica_score_ == defaults.replica_score_);
  }
  // Malformed YAML: best-effort parsing keeps the defaults, no throw.
  {
    clio::run::PoolConfig pool_config;
    pool_config.config_ = "num_replicas: [unclosed\n  ::: garbage";
    ReplicationConfig config;
    ReplicationConfig defaults;
    config.LoadConfig(pool_config);
    REQUIRE(config.num_replicas_ == defaults.num_replicas_);
  }
}

SIMPLE_TEST_MAIN()
