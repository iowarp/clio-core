/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * CHECKPOINT CHIMOD TESTS (vector.Copy fault handler)
 *
 * A tag created WITH a fault handler (checkpoint pool + source tag name in
 * the params) resolves missing blobs lazily from the source. Verifies:
 *   - GET FAULT: reading a blob the copy tag does not hold materialises it
 *     from the source and serves the source's bytes;
 *   - INDEPENDENCE: after materialisation the copy is its own blob -- a
 *     later overwrite of the SOURCE does not change the copy, and a write
 *     to the COPY does not change the source;
 *   - PUT FAULT (materialise-then-merge): a PARTIAL put into a missing
 *     blob first materialises the source content, so the untouched
 *     remainder carries the copied-from bytes, not zeros;
 *   - RANGE READS fault the whole blob (no silent holes);
 *   - SOURCE-ABSENT FALLBACK: a blob the source lacks behaves exactly as
 *     it would on a plain tag (create_on_get semantics preserved).
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/checkpoint/checkpoint_client.h>

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

class CheckpointFaultFixture {
 public:
  std::string config_path_;
  std::string restart_log_path_;

  CheckpointFaultFixture() {
    config_path_ = chi_test_data_dir() + "/checkpoint_fault_config.yaml";
    restart_log_path_ = chi_test_data_dir() + "/checkpoint_fault_restart.bin";
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

  ~CheckpointFaultFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(restart_log_path_)) fs::remove(restart_log_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Checkpoint chimod test configuration
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
      - path: "ram::checkpoint_fault_dram"
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

TEST_CASE("CheckpointFault - lazy copy via tag fault handler",
          "[cte][checkpoint]") {
  CheckpointFaultFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  // checkpoint(565) resolving through core(512).
  {
    clio::cte::checkpoint::Client ckpt(
        clio::cte::checkpoint::kCheckpointPoolId,
        clio::cte::core::kCtePoolId);
    clio::cte::checkpoint::CheckpointConfig params;
    params.next_pool_id_ = clio::cte::core::kCtePoolId;
    auto create = ckpt.AsyncCreateCheckpoint(
        clio::run::PoolQuery::Local(),
        clio::cte::checkpoint::kCheckpointPoolName,
        clio::cte::checkpoint::kCheckpointPoolId, params);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  // Source tag with one full blob.
  const std::string kSrcName = "ckpt_src_tag";
  clio::cte::core::TagId src_tag;
  {
    auto tf = core->AsyncGetOrCreateTag(kSrcName);
    tf.Wait();
    REQUIRE(tf->GetReturnCode() == 0);
    src_tag = tf->tag_id_;
  }
  const std::string v_src = Payload(kValSize, 'a');
  {
    auto put = core->AsyncPutBlob(src_tag, "pg0", 0, kValSize, v_src.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    auto put1 = core->AsyncPutBlob(src_tag, "pg1", 0, kValSize, v_src.data());
    put1.Wait();
    REQUIRE(put1->GetReturnCode() == 0);
  }

  // Copy tag: fault handler = checkpoint pool, params = source tag name.
  clio::cte::core::TagId copy_tag;
  {
    auto tf = core->AsyncGetOrCreateTag(
        "ckpt_copy_tag", clio::cte::checkpoint::kCheckpointPoolId,
        clio::cte::checkpoint::kCheckpointPoolName, kSrcName);
    tf.Wait();
    REQUIRE(tf->GetReturnCode() == 0);
    copy_tag = tf->tag_id_;
  }

  // ======================================================================
  // 1. GET FAULT: the copy holds nothing, yet reads serve source bytes.
  // ======================================================================
  {
    std::vector<char> got(kValSize, 0);
    auto get = core->AsyncGetBlob(copy_tag, "pg0", 0, kValSize,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v_src.data(), kValSize) == 0);
    // ... and the blob is now MATERIALISED in the copy tag.
    auto sz = core->AsyncGetBlobSize(copy_tag, "pg0");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    REQUIRE(sz->size_ == kValSize);
  }

  // ======================================================================
  // 2. INDEPENDENCE, source side: overwriting the SOURCE after the copy
  //    materialised must not change the copy (it is a checkpoint).
  // ======================================================================
  {
    const std::string v2 = Payload(kValSize, 'z');
    auto put = core->AsyncPutBlob(src_tag, "pg0", 0, kValSize, v2.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    std::vector<char> got(kValSize, 0);
    auto get = core->AsyncGetBlob(copy_tag, "pg0", 0, kValSize,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v_src.data(), kValSize) == 0);
  }

  // ======================================================================
  // 3. PUT FAULT (materialise-then-merge): a PARTIAL put into the missing
  //    "pg1" first pulls the source content, so the bytes around the put
  //    range are the source's, not zeros. And the SOURCE is untouched.
  // ======================================================================
  {
    const clio::run::u64 kOff = 1024, kLen = 2048;
    const std::string patch(kLen, 'X');
    auto put = core->AsyncPutBlob(copy_tag, "pg1", kOff, kLen, patch.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    std::string want = v_src;
    std::memcpy(&want[kOff], patch.data(), kLen);
    std::vector<char> got(kValSize, 0);
    auto get = core->AsyncGetBlob(copy_tag, "pg1", 0, kValSize,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), want.data(), kValSize) == 0);

    // The source still holds its original pg1.
    std::vector<char> src_got(kValSize, 0);
    auto sget = core->AsyncGetBlob(src_tag, "pg1", 0, kValSize,
                                   /*flags=*/0, src_got.data());
    sget.Wait();
    REQUIRE(sget->GetReturnCode() == 0);
    REQUIRE(std::memcmp(src_got.data(), v_src.data(), kValSize) == 0);
  }

  // ======================================================================
  // 4. RANGE READ faults the WHOLE blob: ask for a middle slice of a blob
  //    the copy does not hold; the slice is right AND the materialised
  //    blob is full-size (no interval-cannot-describe-a-hole corruption).
  // ======================================================================
  {
    const std::string kSlice = "pg_slice";
    auto sput = core->AsyncPutBlob(src_tag, kSlice.c_str(), 0, kValSize,
                                   v_src.data());
    sput.Wait();
    REQUIRE(sput->GetReturnCode() == 0);

    const clio::run::u64 kOff = 4096, kLen = 512;
    std::vector<char> got(kLen, 0);
    auto get = core->AsyncGetBlob(copy_tag, kSlice.c_str(), kOff, kLen,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), v_src.data() + kOff, kLen) == 0);

    auto sz = core->AsyncGetBlobSize(copy_tag, kSlice.c_str());
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    REQUIRE(sz->size_ == kValSize);
  }

  // ======================================================================
  // 5. SOURCE-ABSENT FALLBACK: a blob the source lacks keeps the plain
  //    tag's semantics -- an ordinary read fails, a create_on_get read
  //    succeeds with the buffer untouched (the pager contract).
  // ======================================================================
  {
    std::vector<char> got(kValSize, 0x5A);
    auto get = core->AsyncGetBlob(copy_tag, "pg_absent", 0, kValSize,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() != 0);

    clio::cte::core::Context cg;
    cg.create_on_get_ = true;
    auto get2 = core->AsyncGetBlob(copy_tag, "pg_absent2", 0, kValSize,
                                   /*flags=*/0, got.data(),
                                   clio::run::PoolQuery::Dynamic(), cg);
    get2.Wait();
    REQUIRE(get2->GetReturnCode() == 0);
    for (clio::run::u64 i = 0; i < kValSize; ++i) {
      REQUIRE(got[i] == 0x5A);
      if (got[i] != 0x5A) break;
    }
  }

  std::printf("checkpoint fault-handler tests passed\n");
}

SIMPLE_TEST_MAIN()
