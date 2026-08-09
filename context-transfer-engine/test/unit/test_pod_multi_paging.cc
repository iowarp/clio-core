/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * BATCHED POD PAGING TESTS
 *
 * PodMultiPutBlob / PodMultiGetBlob / PodMultiScore carry up to kPodMultiMax
 * page requests in ONE task, so a gpu_vector block can flush its whole page
 * cache in a handful of submissions instead of one per page. The batch is only
 * worth anything if a batched page is indistinguishable from a scalar one, so
 * that is what these tests pin down: identical bytes, per-record return codes,
 * and the same behavior through the compressor interposer.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_ctp/compress/compress_factory.h>

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

static const clio::run::PoolId kCompPoolId(563, 0);
static constexpr clio::run::u64 kPageBytes = 16 * 1024;

class PodMultiFixture {
 public:
  std::string config_path_;

  PodMultiFixture() {
    config_path_ = chi_test_data_dir() + "/pod_multi_paging_config.yaml";
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

  ~PodMultiFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# Batched POD paging test configuration - RAM only
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
      - path: "ram::pod_multi_dram"
        bdev_type: "ram"
        capacity_limit: "128MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }
};

namespace {

/** Page `idx` filled with a byte pattern derived from `idx`. */
std::vector<char> PagePattern(clio::run::u32 idx, clio::run::u64 size) {
  std::vector<char> v(size);
  for (clio::run::u64 i = 0; i < size; ++i) {
    v[i] = static_cast<char>((idx * 7 + i * 31) & 0xFF);
  }
  return v;
}

/** Highly compressible page — a codec must actually shrink this. */
std::vector<char> ZeroPage(clio::run::u64 size) {
  return std::vector<char>(size, 0);
}

/** A codec wire id this build actually has (codecs are individually optional;
 *  a hardcoded id makes the test depend on the host's dev packages). */
int AvailableCompressLib() {
  for (const char *name : {"lz4", "zstd", "zlib", "lzma", "bzip2", "brotli",
                           "snappy", "blosc2"}) {
    if (ctp::CompressionFactory::GetPreset(name) != nullptr) {
      return ctp::CompressionFactory::GetWireId(name);
    }
  }
  return 0;
}

/** SHM buffer holding `src`, in the form the POD tasks take. */
struct ShmPage {
  ctp::ipc::FullPtr<char> buf;
  explicit ShmPage(const std::vector<char> &src) {
    buf = CLIO_CPU_IPC->AllocateBuffer(src.size());
    REQUIRE(!buf.IsNull());
    std::memcpy(buf.ptr_, src.data(), src.size());
  }
  ShmPage(const ShmPage &) = delete;
  ShmPage &operator=(const ShmPage &) = delete;
  ~ShmPage() { CLIO_CPU_IPC->FreeBuffer(buf); }
  ctp::ipc::ShmPtr<> Shm() const { return buf.shm_.template Cast<void>(); }
};

}  // namespace

TEST_CASE("PodMultiPaging - batched put/get round-trips byte-identically",
          "[cte][pod][batch]") {
  PodMultiFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  REQUIRE(core != nullptr);

  clio::cte::core::Tag tag("pod_multi_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const clio::run::u32 kPages = 8;

  std::vector<std::vector<char>> want;
  std::vector<std::unique_ptr<ShmPage>> src;
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    want.push_back(PagePattern(i, kPageBytes));
    src.push_back(std::make_unique<ShmPage>(want.back()));
  }

  // Batched put: one task, kPages records.
  {
    auto task = core->NewPodBatch<clio::cte::core::PodMultiPutBlobTask>(
        tag_id, clio::cte::core::Context(), /*flags=*/0,
        clio::run::PoolQuery::Local());
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      std::string name = "pg" + std::to_string(i);
      REQUIRE(task.get()->Add(name.c_str(), 0, kPageBytes, src[i]->Shm(),
                              /*score=*/1.0f));
    }
    REQUIRE(task.get()->count_ == kPages);
    auto fut = core->AsyncPodBatch(task);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->num_ok_ == kPages);
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      REQUIRE(fut->reqs_[i].rc_ == 0);
    }
  }

  // Every page must be readable by an ORDINARY scalar get: a batched put is a
  // put, not a private format.
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    std::vector<char> got(kPageBytes, 0);
    auto get = core->AsyncGetBlob(tag_id, "pg" + std::to_string(i), 0,
                                  kPageBytes, /*flags=*/0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    REQUIRE(std::memcmp(got.data(), want[i].data(), kPageBytes) == 0);
  }

  // Batched get into fresh buffers.
  {
    std::vector<std::unique_ptr<ShmPage>> dst;
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      dst.push_back(std::make_unique<ShmPage>(ZeroPage(kPageBytes)));
    }
    auto task = core->NewPodBatch<clio::cte::core::PodMultiGetBlobTask>(
        tag_id, clio::cte::core::Context(), /*flags=*/0,
        clio::run::PoolQuery::Local());
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      std::string name = "pg" + std::to_string(i);
      REQUIRE(task.get()->Add(name.c_str(), 0, kPageBytes, dst[i]->Shm()));
    }
    auto fut = core->AsyncPodBatch(task);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->num_ok_ == kPages);
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      REQUIRE(fut->reqs_[i].rc_ == 0);
      REQUIRE(std::memcmp(dst[i]->buf.ptr_, want[i].data(), kPageBytes) == 0);
    }
  }
}

TEST_CASE("PodMultiPaging - a full batch is exactly kPodMultiMax records",
          "[cte][pod][batch]") {
  PodMultiFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag("pod_multi_cap_tag");

  auto task = core->NewPodBatch<clio::cte::core::PodMultiPutBlobTask>(
      tag.GetTagId(), clio::cte::core::Context(), 0,
      clio::run::PoolQuery::Local());
  for (clio::run::u32 i = 0; i < clio::cte::core::kPodMultiMax; ++i) {
    REQUIRE(task.get()->Add("x", 0, 0, ctp::ipc::ShmPtr<>::GetNull()));
  }
  // Full: the caller's signal to submit this batch and start another. A block
  // with 256 pages therefore flushes in 256/kPodMultiMax submissions.
  REQUIRE(!task.get()->Add("x", 0, 0, ctp::ipc::ShmPtr<>::GetNull()));
  REQUIRE(task.get()->count_ == clio::cte::core::kPodMultiMax);
}

TEST_CASE("PodMultiPaging - one bad record fails alone",
          "[cte][pod][batch]") {
  PodMultiFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag("pod_multi_partial_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();

  auto good = PagePattern(3, kPageBytes);
  ShmPage src(good);
  {
    auto put = core->AsyncPodPutBlob(tag_id, "present", 0, kPageBytes,
                                     src.Shm(), -1.0f,
                                     clio::cte::core::Context(), 0,
                                     clio::run::PoolQuery::Local());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }

  ShmPage d0(ZeroPage(kPageBytes));
  ShmPage d1(ZeroPage(kPageBytes));
  auto task = core->NewPodBatch<clio::cte::core::PodMultiGetBlobTask>(
      tag_id, clio::cte::core::Context(), 0, clio::run::PoolQuery::Local());
  REQUIRE(task.get()->Add("present", 0, kPageBytes, d0.Shm()));
  REQUIRE(task.get()->Add("missing", 0, kPageBytes, d1.Shm()));
  auto fut = core->AsyncPodBatch(task);
  fut.Wait();
  // The batch reports the failure, but the good record still ran and landed.
  REQUIRE(fut->GetReturnCode() != 0);
  REQUIRE(fut->num_ok_ == 1);
  REQUIRE(fut->reqs_[0].rc_ == 0);
  REQUIRE(fut->reqs_[1].rc_ != 0);
  REQUIRE(std::memcmp(d0.buf.ptr_, good.data(), kPageBytes) == 0);
}

TEST_CASE("PodMultiPaging - batched rescore", "[cte][pod][batch]") {
  PodMultiFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag("pod_multi_score_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const clio::run::u32 kPages = 4;

  std::vector<std::unique_ptr<ShmPage>> src;
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    src.push_back(std::make_unique<ShmPage>(PagePattern(i, kPageBytes)));
    auto put = core->AsyncPodPutBlob(
        tag_id, ("sp" + std::to_string(i)).c_str(), 0, kPageBytes,
        src.back()->Shm(), -1.0f, clio::cte::core::Context(), 0,
        clio::run::PoolQuery::Local());
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
  }

  auto task = core->NewPodBatch<clio::cte::core::PodMultiScoreTask>(
      tag_id, clio::cte::core::Context(), 0, clio::run::PoolQuery::Local());
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    std::string name = "sp" + std::to_string(i);
    REQUIRE(task.get()->Add(name.c_str(), 0, 0, ctp::ipc::ShmPtr<>::GetNull(),
                            /*score=*/0.25f * static_cast<float>(i + 1)));
  }
  auto fut = core->AsyncPodBatch(task);
  fut.Wait();
  REQUIRE(fut->GetReturnCode() == 0);
  REQUIRE(fut->num_ok_ == kPages);

  // Rescoring must not disturb the data.
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    std::vector<char> got(kPageBytes, 0);
    auto get = core->AsyncGetBlob(tag_id, "sp" + std::to_string(i), 0,
                                  kPageBytes, 0, got.data());
    get.Wait();
    REQUIRE(get->GetReturnCode() == 0);
    auto want = PagePattern(i, kPageBytes);
    REQUIRE(std::memcmp(got.data(), want.data(), kPageBytes) == 0);
  }
}

TEST_CASE("PodMultiPaging - batches compress through the interposer",
          "[cte][pod][batch][compressor]") {
  PodMultiFixture fixture;
  auto *core = CLIO_CTE_CLIENT;
  const int lib = AvailableCompressLib();
  REQUIRE(lib != 0);

  {
    clio::cte::compressor::Client comp(kCompPoolId, clio::cte::core::kCtePoolId);
    auto create = comp.AsyncCreateCompressor(clio::run::PoolQuery::Local(),
                                             "clio_cte_pod_multi_comp",
                                             kCompPoolId);
    create.Wait();
    REQUIRE(create->GetReturnCode() == 0);
  }
  clio::cte::core::Client comp_io(kCompPoolId);

  clio::cte::core::Tag tag("pod_multi_comp_tag");
  const clio::cte::core::TagId tag_id = tag.GetTagId();
  const clio::run::u32 kPages = 6;

  clio::cte::core::Context ctx;
  ctx.compress_lib_ = lib;

  // Zero pages: the codec must genuinely shrink them, which is how the raw
  // core view below proves the batch really went through the compressor.
  std::vector<std::unique_ptr<ShmPage>> src;
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    src.push_back(std::make_unique<ShmPage>(ZeroPage(kPageBytes)));
  }
  {
    auto task = comp_io.NewPodBatch<clio::cte::core::PodMultiPutBlobTask>(
        tag_id, ctx, 0, clio::run::PoolQuery::Local());
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      std::string name = "cz" + std::to_string(i);
      REQUIRE(task.get()->Add(name.c_str(), 0, kPageBytes, src[i]->Shm(), 1.0f));
    }
    auto fut = comp_io.AsyncPodBatch(task);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->num_ok_ == kPages);
  }

  // Stored form is smaller than logical — it was compressed, not forwarded.
  for (clio::run::u32 i = 0; i < kPages; ++i) {
    auto raw = core->AsyncGetBlobSize(tag_id, "cz" + std::to_string(i));
    raw.Wait();
    REQUIRE(raw->GetReturnCode() == 0);
    REQUIRE(raw->size_ > 0);
    REQUIRE(raw->size_ < kPageBytes);
  }

  // Batched get through the compressor returns the LOGICAL bytes.
  {
    std::vector<std::unique_ptr<ShmPage>> dst;
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      dst.push_back(std::make_unique<ShmPage>(
          std::vector<char>(kPageBytes, static_cast<char>(0xEE))));
    }
    auto task = comp_io.NewPodBatch<clio::cte::core::PodMultiGetBlobTask>(
        tag_id, ctx, 0, clio::run::PoolQuery::Local());
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      std::string name = "cz" + std::to_string(i);
      REQUIRE(task.get()->Add(name.c_str(), 0, kPageBytes, dst[i]->Shm()));
    }
    auto fut = comp_io.AsyncPodBatch(task);
    fut.Wait();
    REQUIRE(fut->GetReturnCode() == 0);
    REQUIRE(fut->num_ok_ == kPages);
    const std::vector<char> zeros = ZeroPage(kPageBytes);
    for (clio::run::u32 i = 0; i < kPages; ++i) {
      REQUIRE(fut->reqs_[i].rc_ == 0);
      REQUIRE(std::memcmp(dst[i]->buf.ptr_, zeros.data(), kPageBytes) == 0);
    }
  }
}

SIMPLE_TEST_MAIN()
