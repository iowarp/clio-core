/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * NeuroPress via compose YAML (issue #693).
 *
 * test_compressor_compose_config proves the compose keys reach
 * CompressorConfig. This proves the next link: that a compressor composed
 * from that YAML actually LOADS the model and selects with it.
 *
 * The distinction matters because it is exactly where the integration was
 * broken. Every NeuroPress example and every existing test sets
 * `cfg.neuropress_model_path_` programmatically in C++ -- ten out of ten of
 * them -- so the engine was thoroughly exercised while the one route a
 * DEPLOYMENT takes, a config file, was never covered. LoadConfig silently
 * dropped the key and nothing noticed.
 *
 * The observable is the selection itself. DynamicSchedule returns its
 * Context with compress_lib_ set to what was actually chosen, so the test
 * asserts the winner comes from NeuroPress's trained action space -- the
 * eight GPU-lossless nvcomp codecs -- rather than the CPU default a
 * compressor with no model falls back to. That cannot be satisfied by
 * accident: nothing else in the compressor ranks those eight.
 *
 * The compose file is generated here rather than checked in so the weights
 * path is the build's own, which also means the YAML under test is written
 * the same way a Jarvis pipeline writes it.
 */

#include "simple_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/core/blob_transform.h>
#include <clio_ctp/compress/compress_factory.h>

#if CTP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

bool g_initialized = false;
fs::path g_conf_path;

/** True when a GPU is actually present; NeuroPress inference needs one. */
bool HaveGpu() {
#if CTP_ENABLE_CUDA
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
  return false;
#endif
}

/**
 * Write the compose file: compressor at the CTE entrypoint (512.0) with the
 * NeuroPress model configured, core behind it at 513.0. Same shape as
 * test_transparent_compress_config.yaml, plus the keys under test.
 */
fs::path WriteComposeConfig() {
  fs::path dir = fs::temp_directory_path() / "clio_np_compose_test";
  fs::create_directories(dir);
  fs::path path = dir / "neuropress_compose.yaml";

  std::ofstream f(path);
  f << "networking:\n"
    << "  port: 9413\n"
    << "runtime:\n"
    << "  num_threads: 4\n"
    << "  queue_depth: 1024\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n"
    << "    pool_name: \"ram::chi_default_bdev\"\n"
    << "    pool_query: local\n"
    << "    pool_id: \"301.0\"\n"
    << "    bdev_type: ram\n"
    << "    capacity: \"256MB\"\n"
    << "  - mod_name: clio_cte_compressor\n"
    << "    pool_name: cte_compressor\n"
    << "    pool_query: local\n"
    << "    pool_id: \"512.0\"\n"
    << "    next_pool_id: \"513.0\"\n"
    // The key this whole test exists for. Inference only: online learning
    // and exploration stay off, matching upstream's defaults, so a run that
    // merely points at weights does not silently start training.
    << "    neuropress_model_path: \"" << CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
    << "\"\n"
    << "  - mod_name: clio_cte_core\n"
    << "    pool_name: cte_core\n"
    << "    pool_query: local\n"
    << "    pool_id: \"513.0\"\n"
    << "    storage:\n"
    << "      - path: \"ram::cte_ram_tier1\"\n"
    << "        bdev_type: \"ram\"\n"
    << "        capacity_limit: \"128MB\"\n"
    << "        score: 1.0\n"
    << "    dpe:\n"
    << "      dpe_type: \"max_bw\"\n";
  f.close();
  return path;
}

void EnsureInit() {
  if (g_initialized) return;
  g_conf_path = WriteComposeConfig();
  SIMPLE_TEST_SETENV("CLIO_SERVER_CONF", g_conf_path.string().c_str());

  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  g_initialized = true;
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(1s);

  // Compose already created every pool; bind straight to the entrypoint,
  // which is the compressor.
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);
}

/** NeuroPress ranks only the eight GPU-lossless nvcomp codecs. */
bool IsNeuroPressAction(int wire_id) {
  const std::string name = ctp::CompressionFactory::NameForWireId(wire_id);
  return name.rfind("nvcomp-", 0) == 0;
}

}  // namespace

TEST_CASE("Compose YAML alone turns NeuroPress selection on",
          "[compressor][neuropress][compose][693]") {
  if (!HaveGpu()) {
    INFO("No CUDA device -- skipping NeuroPress compose test");
    return;
  }
  EnsureInit();

  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag_task = cte_client->AsyncGetOrCreateTag("np_compose_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  // Smooth float data: compressible, and float is what the model was
  // trained on. A pattern of bytes would rank differently and prove less.
  const size_t kElems = 256 * 1024;
  const size_t kBytes = kElems * sizeof(float);
  auto buffer = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!buffer.IsNull());
  auto *f = reinterpret_cast<float *>(buffer.ptr_);
  for (size_t i = 0; i < kElems; ++i) {
    f[i] = static_cast<float>((i % 512) * 0.015625);
  }
  ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();

  clio::cte::core::Context ctx;
#if CTP_ENABLE_COMPRESS
  // dynamic_compress_: 0 = skip, 1 = STATIC (caller pins compress_lib_),
  // 2 = DYNAMIC (the model chooses). It must be 2 -- the NeuroPress gate in
  // EstCompressionStats is `dynamic_compress_ != 1`, so the static value
  // silently bypasses the model and falls through to the legacy heuristics.
  ctx.dynamic_compress_ = 2;
  ctx.data_type_ = 1;  // float32 -- what the model was trained on
#endif

  clio::cte::compressor::Client compressor(clio::cte::core::kCtePoolId);
  auto sched = compressor.AsyncDynamicSchedule(
      clio::run::PoolQuery::Local(), tag_id, "np_compose_blob",
      /*offset=*/0, kBytes, blob_data, -1.0f, ctx, 0, cte_client->pool_id_);
  sched.Wait();
  REQUIRE(sched->GetReturnCode() == 0);

  const int chosen = sched->context_.compress_lib_;
  const std::string chosen_name =
      ctp::CompressionFactory::NameForWireId(chosen);
  INFO("NeuroPress chose wire id " + std::to_string(chosen) + " (" +
       chosen_name + ")");

  // The assertion that distinguishes a loaded model from a silently absent
  // one: only the NeuroPress ranker produces these eight.
  REQUIRE(IsNeuroPressAction(chosen));
  REQUIRE(sched->context_.actual_compression_ratio_ > 1.0f);
}

TEST_CASE("Composed NeuroPress compressor round-trips its data",
          "[compressor][neuropress][compose][693]") {
  if (!HaveGpu()) {
    INFO("No CUDA device -- skipping NeuroPress compose round trip");
    return;
  }
  EnsureInit();

  // Selecting well is worthless if the bytes do not come back. Reads go
  // through the same entrypoint and must decompress whatever the model
  // picked, without the caller naming the codec.
  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag_task = cte_client->AsyncGetOrCreateTag("np_compose_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  const size_t kElems = 256 * 1024;
  const size_t kBytes = kElems * sizeof(float);
  auto get_buf = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!get_buf.IsNull());
  std::memset(get_buf.ptr_, 0, kBytes);
  ctp::ipc::ShmPtr<> get_data = get_buf.shm_.template Cast<void>();

  // Ask the CORE pool (513.0) directly what it is physically holding. The
  // compressor's own GetBlobSize is overridden to report the LOGICAL size,
  // so it cannot answer this question -- it returns 1 MiB either way.
  //
  // Do NOT assert on context_.transform_flags_ here: Runtime::GetBlob
  // deliberately CLEARS kBlobTransformCompressed on the way out, because by
  // then the caller holds original bytes and nothing downstream should try
  // to undo the codec again. A zero there is the success contract, not
  // evidence of an uncompressed blob.
  {
    clio::cte::core::Client core(clio::run::PoolId(513, 0));
    auto sz = core.AsyncGetBlobSize(tag_id, "np_compose_blob");
    sz.Wait();
    REQUIRE(sz->GetReturnCode() == 0);
    INFO("core-stored size " + std::to_string(sz->size_) + " B vs original " +
         std::to_string(kBytes) + " B");
    // Strictly smaller: Compress only keeps codec output when it beats the
    // input, so this is exactly "the stored bytes are compressed".
    REQUIRE(sz->size_ < kBytes);
  }

  auto get_task = cte_client->AsyncGetBlob(tag_id, "np_compose_blob", 0,
                                           kBytes, 0, get_data);
  get_task.Wait();
  REQUIRE(get_task->GetReturnCode() == 0);

  auto *got = reinterpret_cast<float *>(get_buf.ptr_);
  size_t mismatches = 0;
  for (size_t i = 0; i < kElems; ++i) {
    if (got[i] != static_cast<float>((i % 512) * 0.015625)) ++mismatches;
  }
  INFO(std::to_string(mismatches) + " / " + std::to_string(kElems) +
       " elements differ");
  REQUIRE(mismatches == 0);
}

SIMPLE_TEST_MAIN()
