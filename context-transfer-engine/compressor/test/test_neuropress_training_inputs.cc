/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * The online update must be shown the config that actually ran. Drives the
 * real compressor and asserts on NeuroPressChunkDiag that feat_eb_enc is the
 * configured bound, feat_action == nn_original_action (what was predicted),
 * and nn_action matches the preset the blob was stored under.
 */
#include "simple_test.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/content_transfer_engine.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/compressor/neuropress_chunk_diag.h>
#include <clio_ctp/compress/compress_factory.h>

#if CTP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

/** Nonzero so the quantize half is selectable and the field under test is
 *  not legitimately 0. */
constexpr double kErrorBound = 1e-3;

bool HaveGpu() {
#if CTP_ENABLE_CUDA
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
  return false;
#endif
}

/** Bit 24 of compress_preset_. Spelled out, not reusing the compressor's
 *  UnpackQuantEnabled, which could not catch it packing the bit elsewhere. */
bool QuantizeBitOf(uint32_t packed_preset) {
  return ((packed_preset >> 24) & 1u) != 0;
}

/** Set by ctest. Only with exploration on can the written action differ from
 *  the predicted one, which is what makes invariant 2 falsifiable. */
bool ExplorationRequested() {
  const char *v = std::getenv("NP_TEST_EXPLORATION");
  return v != nullptr && v[0] == '1';
}

/* MAPE gate at 0, so training runs on every chunk that mispredicts. */
std::string WriteComposeConfig(const std::string &path, bool exploration) {
  std::ofstream f(path);
  f << "networking:\n  port: 9413\n"
    << "runtime:\n  num_threads: 4\n  queue_depth: 1024\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
    << "    pool_query: local\n    pool_id: \"301.0\"\n"
    << "    bdev_type: ram\n    capacity: \"256MB\"\n"
    << "  - mod_name: clio_cte_compressor\n    pool_name: cte_compressor\n"
    << "    pool_query: local\n    pool_id: \"512.0\"\n"
    << "    next_pool_id: \"513.0\"\n"
    << "    neuropress_model_path: \"" << CLIO_CTP_NEUROPRESS_WEIGHTS_DIR << "\"\n"
    << "    neuropress_online_learning_enabled: true\n"
    << "    neuropress_mape_threshold: 0.0\n"
    << "    neuropress_exploration_enabled: "
    << (exploration ? "true" : "false") << "\n"
    // Threshold 0 and k=31 so an alternative genuinely can win.
    << (exploration ? "    neuropress_exploration_threshold: 0.0\n"
                      "    neuropress_exploration_k: 31\n"
                    : "")
    << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
    << "    pool_query: local\n    pool_id: \"513.0\"\n"
    << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
    << "        bdev_type: \"ram\"\n        capacity_limit: \"128MB\"\n"
    << "        score: 1.0\n"
    << "    dpe:\n      dpe_type: \"max_bw\"\n";
  f.close();
  return path;
}

/** Distinct shapes, so the selection is not one action repeated. */
void FillChunk(float *fp, size_t n, int shape) {
  for (size_t i = 0; i < n; ++i) {
    switch (shape) {
      case 0:
        fp[i] = std::sin(static_cast<double>(i) * 0.001) * 100.0;
        break;
      case 1:
        fp[i] = static_cast<float>((i / 512) % 1024) * 0.25f;
        break;
      case 2: {
        uint32_t x = static_cast<uint32_t>(i) * 2654435761u + 0x9E3779B9u;
        x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13;
        fp[i] = static_cast<float>(x) / 4294967295.0f;
        break;
      }
      default:
        fp[i] = 42.0f;
        break;
    }
  }
}

}  // namespace

TEST_CASE("Online learning is shown the config that actually ran",
          "[compressor][neuropress][learning][693]") {
  if (!HaveGpu()) {
    INFO("No CUDA device -- skipping");
    return;
  }
  const bool exploration = ExplorationRequested();
  // Parenthesized: INFO streams, and << binds tighter than ?:.
  INFO((exploration ? "exploration ON" : "exploration OFF"));
  const std::string cfg = WriteComposeConfig(
      exploration ? "/tmp/np_training_inputs_explore.yaml"
                  : "/tmp/np_training_inputs.yaml",
      exploration);
  SIMPLE_TEST_SETENV("CLIO_SERVER_CONF", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);

  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag_task = cte_client->AsyncGetOrCreateTag("np_training_inputs_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  clio::cte::compressor::NeuroPressResetChunkHistory();
  REQUIRE(clio::cte::compressor::NeuroPressChunkHistoryCount() == 0);

  const size_t kElems = 256 * 1024;
  const size_t kBytes = kElems * sizeof(float);
  constexpr int kShapes = 4;

  // What each blob was stored under, for cross-checking the record.
  std::vector<uint32_t> stored_preset;

  for (int shape = 0; shape < kShapes; ++shape) {
    auto buf = CLIO_IPC->AllocateBuffer(kBytes);
    REQUIRE(!buf.IsNull());
    FillChunk(reinterpret_cast<float *>(buf.ptr_), kElems, shape);

    clio::cte::core::Context ctx;
    ctx.dynamic_compress_ = 2;  // DYNAMIC: let the model choose
    ctx.data_type_ = 1;         // float32
    ctx.error_bound_ = kErrorBound;

    clio::cte::compressor::Client compressor(clio::cte::core::kCtePoolId);
    auto sched = compressor.AsyncDynamicSchedule(
        clio::run::PoolQuery::Local(), tag_id,
        "np_train_blob_" + std::to_string(shape),
        /*offset=*/0, kBytes, buf.shm_.template Cast<void>(), -1.0f, ctx, 0,
        clio::run::PoolId(513, 0));
    sched.Wait();
    REQUIRE(sched->GetReturnCode() == 0);
    stored_preset.push_back(
        static_cast<uint32_t>(sched->context_.compress_preset_));
  }

  const int n = clio::cte::compressor::NeuroPressChunkHistoryCount();
  INFO(std::to_string(n) + " chunk record(s) for " + std::to_string(kShapes) +
       " write(s)");
  // Empty history = nothing asserted, so a failure rather than a skip.
  REQUIRE(n > 0);
  REQUIRE(n <= kShapes);

  int quantize_actions = 0, lossless_actions = 0, trained = 0;
  int explored = 0, swapped = 0;

  for (int i = 0; i < n; ++i) {
    clio::cte::compressor::NeuroPressChunkDiag d;
    REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(i, &d) == 0);

    INFO("chunk " + std::to_string(i) + ": feat_action=" +
         std::to_string(d.feat_action) + " orig=" +
         std::to_string(d.nn_original_action) + " final=" +
         std::to_string(d.nn_action) + " explored=" +
         std::to_string(d.exploration_triggered) + " eb=" +
         std::to_string(d.feat_eb_enc) + " size=" +
         std::to_string(d.feat_ds_enc) + " H=" +
         std::to_string(d.feat_entropy) + " regret=" +
         std::to_string(d.regret) + " sgd=" +
         std::to_string(d.sgd_fired));

    // 1: the configured bound reached the model -- the field that was 0.
    // Every chunk: upstream passes cfg.error_bound whichever action won.
    REQUIRE(static_cast<double>(d.feat_eb_enc) ==
            static_cast<double>(static_cast<float>(kErrorBound)));

    // 2: feat_action follows the PREDICTED config. Upstream sets it from
    // nn_original_action, never nn_action (gpucompress_diagnostics.cpp).
    REQUIRE(d.feat_action >= 0);
    REQUIRE(d.feat_action < 32);
    REQUIRE(d.feat_action == d.nn_original_action);

    // 3: nn_action describes what was WRITTEN. Cross-checked against the
    // stored preset -- two independently produced values.
    REQUIRE(d.nn_action >= 0);
    REQUIRE(d.nn_action < 32);
    const bool final_quantizes = ((d.nn_action / 8) % 2) != 0;
    if (static_cast<size_t>(i) < stored_preset.size()) {
      REQUIRE(final_quantizes == QuantizeBitOf(stored_preset[i]));
    }
    if (final_quantizes) {
      ++quantize_actions;
    } else {
      ++lossless_actions;
    }

    // -1 sentinel unless exploration ran; never -1 meaning "zero".
    if (d.exploration_triggered) {
      ++explored;
      REQUIRE(std::isfinite(d.regret));
      if (d.nn_action != d.nn_original_action) ++swapped;
    } else {
      REQUIRE(static_cast<double>(d.regret) == -1.0);
      // Nothing could have replaced the primary.
      REQUIRE(d.nn_action == d.nn_original_action);
    }

    // Zeroed statistics would train toward a chunk that does not exist.
    REQUIRE(static_cast<double>(d.feat_ds_enc) == static_cast<double>(kBytes));
    REQUIRE(std::isfinite(d.feat_entropy));
    REQUIRE(std::isfinite(d.feat_mad));
    REQUIRE(std::isfinite(d.feat_deriv));
    REQUIRE(d.feat_entropy >= 0.0f);
    REQUIRE(d.feat_entropy <= 8.0f);

    if (d.sgd_fired) ++trained;
  }

  INFO(std::to_string(quantize_actions) + " quantize action(s), " +
       std::to_string(lossless_actions) + " lossless, " +
       std::to_string(trained) + " chunk(s) trained, " +
       std::to_string(explored) + " explored, " + std::to_string(swapped) +
       " swapped by exploration");

  // Must actually have run, or 2 and 3 never see a case that can disagree.
  if (exploration) {
    REQUIRE(explored > 0);
  } else {
    REQUIRE(explored == 0);
  }

  // At least one, not all: costs are clamped first (ratio capped at 100x,
  // times floored 1 ms), so a chunk that saturates gives a MAPE of exactly
  // zero and no update. The constant-valued chunk here does that.
  REQUIRE(trained > 0);
}

SIMPLE_TEST_MAIN()
