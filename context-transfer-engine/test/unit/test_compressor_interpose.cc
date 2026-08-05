/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * COMPRESSOR INTERPOSITION TESTS (issue #886)
 *
 * The compressor chimod speaks the CTE core's task interface: a PLAIN
 * clio::cte::core::Client pointed at the compressor pool gets transparent
 * compression on whole-blob puts (Context::compress_lib_ set) and
 * transparent decompression on reads — including PARTIAL and VECTORED reads
 * of compressed blobs, which the raw core cannot serve (it returns stored
 * codec bytes). GetBlobSize reports the LOGICAL size. Everything else
 * (vectored/partial puts, MultiPutBlob batches, metadata verbs) passes
 * through with the core's semantics.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/compressor/compressor_client.h>

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

static const clio::run::PoolId kCompPoolId(562, 0);
static constexpr clio::run::u64 kValSize = 64 * 1024;

class CompressorInterposeFixture {
 public:
  std::string config_path_;

  CompressorInterposeFixture() {
    config_path_ = chi_test_data_dir() + "/compressor_interpose_config.yaml";
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

  ~CompressorInterposeFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Compressor interposition test configuration - RAM only
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
      - path: "ram::compressor_interpose_dram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0

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

TEST_CASE("CompressorInterpose - transparent compress/decompress + parity",
          "[cte][compressor][interpose][886]") {
  CompressorInterposeFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);
  auto *ipc = CLIO_CPU_IPC;

  // Create the compressor pool over the default core pool.
  {
    clio::cte::compressor::Client comp(kCompPoolId, clio::cte::core::kCtePoolId);
    auto create = comp.AsyncCreateCompressor(clio::run::PoolQuery::Local(),
                                             "clio_cte_compressor_pool",
                                             kCompPoolId);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }

  // The whole point: a PLAIN core client pointed at the compressor pool.
  clio::cte::core::Client comp_io(kCompPoolId);

  clio::cte::core::Tag tag("comp_interpose_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string val = CompressiblePayload(kValSize);

  // A. Uncompressed passthrough: no codec requested — put/get/size behave
  //    exactly like the core.
  {
    auto put = comp_io.AsyncPutBlob(tag_id, "raw_blob", 0, kValSize,
                                    val.data());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    std::vector<char> got(kValSize, 0);
    auto get = comp_io.AsyncGetBlob(tag_id, "raw_blob", 0, kValSize,
                                    /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
    auto sz = comp_io.AsyncGetBlobSize(tag_id, "raw_blob");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    REQUIRE(sz->size_ == kValSize);
  }

  // B. Compressed whole-blob put: the interposer compresses; the RAW core
  //    view proves it (stored < logical, transformed bytes), while every
  //    interposed view stays logical.
  {
    clio::cte::core::Context ctx;
    ctx.compress_lib_ = 1;  // wire id -> factory registry (zstd family)
    auto put = comp_io.AsyncPutBlob(tag_id, "comp_blob", 0, kValSize,
                                    val.data(), /*score=*/-1.0f, ctx);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);

    auto raw_sz = core->AsyncGetBlobSize(tag_id, "comp_blob");
    raw_sz.Wait();
    REQUIRE(raw_sz->GetReturnCode() == 0);
    REQUIRE(raw_sz->size_ > 0);
    REQUIRE(raw_sz->size_ < kValSize);  // genuinely compressed on the core

    auto log_sz = comp_io.AsyncGetBlobSize(tag_id, "comp_blob");
    log_sz.Wait();
    REQUIRE(log_sz->GetReturnCode() == 0);
    REQUIRE(log_sz->size_ == kValSize);  // logical size via the interposer

    std::vector<char> got(kValSize, 0);
    auto get = comp_io.AsyncGetBlob(tag_id, "comp_blob", 0, kValSize,
                                    /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data(), kValSize) == 0);
  }

  // C. Partial + VECTORED reads of the compressed blob — impossible against
  //    the raw core (codec bytes), served correctly through the interposer.
  {
    std::vector<char> part(500, 0);
    auto get = comp_io.AsyncGetBlob(tag_id, "comp_blob", 1000, 500,
                                    /*flags=*/0, part.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(part.data(), val.data() + 1000, 500) == 0);

    ctp::ipc::FullPtr<char> b1 = ipc->AllocateBuffer(256);
    ctp::ipc::FullPtr<char> b2 = ipc->AllocateBuffer(1024);
    REQUIRE(!b1.IsNull());
    REQUIRE(!b2.IsNull());
    std::vector<clio::cte::core::BlobSegment> segs;
    segs.emplace_back(64, 256, ctp::ipc::ShmPtr<>(b1.shm_));
    segs.emplace_back(40000, 1024, ctp::ipc::ShmPtr<>(b2.shm_));
    auto vget = comp_io.AsyncGetBlobVectored(tag_id, "comp_blob", segs);
    vget.Wait();
    REQUIRE(vget->GetReturnCode() == 0);
    REQUIRE(std::memcmp(b1.ptr_, val.data() + 64, 256) == 0);
    REQUIRE(std::memcmp(b2.ptr_, val.data() + 40000, 1024) == 0);
    ipc->FreeBuffer(b1);
    ipc->FreeBuffer(b2);
  }

  // D. Vectored put with a codec REQUESTED stores raw (a partial write
  //    cannot patch a compressed stream) and never records the codec: the
  //    raw core must read the caller's bytes back verbatim.
  {
    ctp::ipc::FullPtr<char> s1 = ipc->AllocateBuffer(4096);
    ctp::ipc::FullPtr<char> s2 = ipc->AllocateBuffer(4096);
    REQUIRE(!s1.IsNull());
    REQUIRE(!s2.IsNull());
    std::memcpy(s1.ptr_, val.data(), 4096);
    std::memcpy(s2.ptr_, val.data() + 8192, 4096);
    std::vector<clio::cte::core::BlobSegment> segs;
    segs.emplace_back(0, 4096, ctp::ipc::ShmPtr<>(s1.shm_));
    segs.emplace_back(8192, 4096, ctp::ipc::ShmPtr<>(s2.shm_));
    clio::cte::core::Context ctx;
    ctx.compress_lib_ = 1;  // requested, but undefined for a vectored put
    auto vput = comp_io.AsyncPutBlobVectored(tag_id, "vec_blob", segs,
                                             /*score=*/-1.0f, ctx);
    vput.Wait();
    REQUIRE(vput->GetReturnCode() == 0);
    ipc->FreeBuffer(s1);
    ipc->FreeBuffer(s2);

    std::vector<char> got(4096, 0);
    auto get = core->AsyncGetBlob(tag_id, "vec_blob", 8192, 4096,
                                  /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), val.data() + 8192, 4096) == 0);
  }

  // E. MultiPutBlob through the compressor pool: the batch executes on the
  //    next pool with the core's exact semantics.
  {
    constexpr int kRecords = 4;
    constexpr clio::run::u64 kRecSize = 2048;
    ctp::ipc::FullPtr<char> staging =
        ipc->AllocateBuffer(kRecords * kRecSize);
    REQUIRE(!staging.IsNull());
    std::vector<clio::cte::core::MultiPutDesc> descs;
    for (int i = 0; i < kRecords; ++i) {
      std::memset(staging.ptr_ + i * kRecSize, 'a' + i, kRecSize);
      clio::cte::core::MultiPutDesc d;
      d.tag_id_ = tag_id;
      d.blob_name_ = "batch_blob_" + std::to_string(i);
      d.offset_ = 0;
      d.size_ = kRecSize;
      d.payload_off_ = i * kRecSize;
      descs.push_back(d);
    }
    auto batch = comp_io.AsyncMultiPutVectored(
        ctp::ipc::ShmPtr<>(staging.shm_), kRecords * kRecSize, descs);
    batch.Wait();
    REQUIRE(batch->GetReturnCode() == 0);
    REQUIRE(batch->num_ok_ == kRecords);
    ipc->FreeBuffer(staging);

    for (int i = 0; i < kRecords; ++i) {
      std::vector<char> got(kRecSize, 0);
      auto get = comp_io.AsyncGetBlob(tag_id, "batch_blob_" + std::to_string(i),
                                      0, kRecSize, /*flags=*/0, got.data());
      get.Wait();
      REQUIRE(get->GetReturnCode() == 0);
      REQUIRE(got[0] == static_cast<char>('a' + i));
      REQUIRE(got[kRecSize - 1] == static_cast<char>('a' + i));
    }
  }

  // F. Metadata verbs forward untouched.
  {
    auto sz = comp_io.AsyncGetTagSize(tag_id);
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
  }
}

SIMPLE_TEST_MAIN()
