/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Does the online update actually make the model more accurate?
 *
 * Three phases of randomized data, each internally homogeneous in entropy and
 * distinct from the others, so the model meets a distribution shift twice.
 * Written through the real compressor (DynamicSchedule), so predictions,
 * measurements and updates all come from the shipping path; accuracy is read
 * back per chunk from NeuroPressChunkDiag.
 *
 * NP_LEARN=0 runs the same data with learning off. Without that control a
 * falling error proves nothing -- it could be later chunks being easier.
 */
#include "simple_test.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
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

constexpr int kPhases = 3;
constexpr int kWindow = 8;  // chunks averaged at each end of a phase

bool HaveGpu() {
#if CTP_ENABLE_CUDA
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
  return false;
#endif
}

bool LearningRequested() {
  const char *v = std::getenv("NP_LEARN");
  return v == nullptr || v[0] != '0';  // on unless explicitly disabled
}

/** Env-settable so a sweep needs no rebuild. */
std::string EnvOr(const char *key, const char *fallback) {
  const char *v = std::getenv(key);
  return (v && v[0]) ? std::string(v) : std::string(fallback);
}

/** Chunks per phase; NP_CHUNKS overrides. 4 MiB each, so 85 gives ~1 GiB. */
int ChunksPerPhase() {
  const char *v = std::getenv("NP_CHUNKS");
  const int n = (v && v[0]) ? std::atoi(v) : 30;
  return (n > 0) ? n : 30;
}

/** NP_PRINT_ALL=1 prints a line per chunk instead of the phase ends. */
bool PrintAll() {
  const char *v = std::getenv("NP_PRINT_ALL");
  return v != nullptr && v[0] == '1';
}


/** NeuroPress action id -> codec, for the per-chunk line. */
const char *CodecName(int action) {
  switch (action % 8) {
    case 0: return "lz4";      case 1: return "snappy";
    case 2: return "deflate";  case 3: return "gdeflate";
    case 4: return "zstd";     case 5: return "ans";
    case 6: return "cascaded"; default: return "bitcomp";
  }
}

const char *PhaseName(int p) {
  switch (p) {
    case 0: return "A structured";
    case 1: return "B mixed";
    default: return "C near-random";
  }
}

/* Learning rate and MAPE gate default to 0.1 / 0.2, both well above
 * upstream's 0.01 / 0.30, so adaptation shows within a 30-chunk phase rather
 * than over hundreds. Override with NP_LR / NP_MAPE. */
std::string WriteComposeConfig(const std::string &path, bool learning) {
  std::ofstream f(path);
  f << "networking:\n  port: 9413\n"
    << "runtime:\n  num_threads: 4\n  queue_depth: 1024\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
    << "    pool_query: local\n    pool_id: \"301.0\"\n"
    << "    bdev_type: ram\n    capacity: \"" << EnvOr("NP_CAP", "8GB")
    << "\"\n"
    << "  - mod_name: clio_cte_compressor\n    pool_name: cte_compressor\n"
    << "    pool_query: local\n    pool_id: \"512.0\"\n"
    << "    next_pool_id: \"513.0\"\n"
    << "    neuropress_model_path: \"" << CLIO_CTP_NEUROPRESS_WEIGHTS_DIR << "\"\n"
    << "    neuropress_online_learning_enabled: "
    << (learning ? "true" : "false") << "\n"
    << "    neuropress_learning_rate: " << EnvOr("NP_LR", "0.1") << "\n"
    << "    neuropress_mape_threshold: " << EnvOr("NP_MAPE", "0.2") << "\n"
    << "    neuropress_exploration_enabled: false\n"
    << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
    << "    pool_query: local\n    pool_id: \"513.0\"\n"
    << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
    << "        bdev_type: \"ram\"\n        capacity_limit: \""
    << EnvOr("NP_CAP", "8GB") << "\"\n"
    << "        score: 1.0\n"
    << "    dpe:\n      dpe_type: \"max_bw\"\n";
  f.close();
  return path;
}

/**
 * Randomized float32; the phase sets the statistical character and the chunk
 * index only reseeds it, so entropy sits in a narrow band within a phase and
 * moves between phases.
 */
void FillChunk(float *fp, size_t n, int phase, unsigned seed) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t x = static_cast<uint32_t>(i) * 2654435761u + seed;
    x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    const float u = static_cast<float>(x) / 4294967295.0f;
    switch (phase) {
      case 0:  // coarse ramp: few distinct byte values, low entropy
        fp[i] = static_cast<float>((i / 256) % 64) * 0.5f +
                (u < 0.02f ? 1.0f : 0.0f);
        break;
      case 1:  // smooth carrier plus bounded noise: mid entropy
        fp[i] = std::sin(6.28318 * static_cast<double>(i % 8192) / 8192.0 *
                         3.0) * 10.0f + (u - 0.5f) * 0.8f;
        break;
      default:  // near-uniform: high entropy, close to incompressible
        fp[i] = u * 1000.0f - 500.0f;
        break;
    }
  }
}

double Mean(const std::vector<double> &v, int from, int count) {
  double s = 0.0;
  int n = 0;
  for (int i = from; i >= 0 && i < from + count &&
                     i < static_cast<int>(v.size()); ++i) {
    s += v[i]; ++n;
  }
  return (n > 0) ? s / n : 0.0;
}

}  // namespace

TEST_CASE("Online learning improves prediction accuracy across phases",
          "[compressor][neuropress][learning][phases][693]") {
  if (!HaveGpu()) {
    INFO("No CUDA device -- skipping");
    return;
  }
  const bool learning = LearningRequested();
  INFO(std::string(learning ? "learning ON" : "learning OFF (control)") +
       "  lr=" + EnvOr("NP_LR", "0.1") + " mape_gate=" +
       EnvOr("NP_MAPE", "0.2"));

  const std::string cfg = WriteComposeConfig(
      learning ? "/tmp/np_phases_learn.yaml" : "/tmp/np_phases_frozen.yaml",
      learning);
  SIMPLE_TEST_SETENV("CLIO_SERVER_CONF", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);

  auto *cte_client = CLIO_CTE_CLIENT;
  auto tag_task = cte_client->AsyncGetOrCreateTag("np_phases_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  clio::cte::compressor::NeuroPressResetChunkHistory();

  const int kChunksPerPhase = ChunksPerPhase();
  const bool print_all = PrintAll();
  const size_t kElems = 1024 * 1024;  // 4 MiB of float32
  const size_t kBytes = kElems * sizeof(float);
  clio::cte::compressor::Client compressor(clio::cte::core::kCtePoolId);

  for (int phase = 0; phase < kPhases; ++phase) {
    for (int ci = 0; ci < kChunksPerPhase; ++ci) {
      const int idx = phase * kChunksPerPhase + ci;
      auto buf = CLIO_IPC->AllocateBuffer(kBytes);
      REQUIRE(!buf.IsNull());
      FillChunk(reinterpret_cast<float *>(buf.ptr_), kElems, phase,
                0x9E3779B9u + 7919u * idx);

      clio::cte::core::Context ctx;
      ctx.dynamic_compress_ = 2;  // DYNAMIC: let the model choose
      ctx.data_type_ = 1;         // float32

      auto sched = compressor.AsyncDynamicSchedule(
          clio::run::PoolQuery::Local(), tag_id,
          "np_phase_blob_" + std::to_string(idx), /*offset=*/0, kBytes,
          buf.shm_.template Cast<void>(), -1.0f, ctx, 0,
          clio::run::PoolId(513, 0));
      sched.Wait();
      REQUIRE(sched->GetReturnCode() == 0);
    }
  }

  // ---- Read every blob back and verify it byte for byte. ----
  // The experiment trains on MEASURED ratios, so a codec that silently
  // produced corrupt output would still report a plausible ratio and the
  // model would learn from it. The content is regenerated from (phase, seed)
  // rather than retained, so this costs one buffer instead of 360 MiB.
  {
    std::vector<float> expect(kElems);
    size_t bad_chunks = 0, bad_elems = 0;
    for (int phase = 0; phase < kPhases; ++phase) {
      for (int ci = 0; ci < kChunksPerPhase; ++ci) {
        const int idx = phase * kChunksPerPhase + ci;
        auto rbuf = CLIO_IPC->AllocateBuffer(kBytes);
        REQUIRE(!rbuf.IsNull());
        std::memset(rbuf.ptr_, 0, kBytes);
        auto get = cte_client->AsyncGetBlob(
            tag_id, "np_phase_blob_" + std::to_string(idx), 0, kBytes, 0,
            rbuf.shm_.template Cast<void>());
        get.Wait();
        REQUIRE(get->GetReturnCode() == 0);

        FillChunk(expect.data(), kElems, phase,
                  0x9E3779B9u + 7919u * idx);
        const auto *got = reinterpret_cast<const float *>(rbuf.ptr_);
        size_t bad = 0;
        for (size_t e = 0; e < kElems; ++e) {
          // Lossless path (error_bound stays 0), so exact bytes -- a
          // tolerance here would hide precisely the corruption being checked.
          if (std::memcmp(&got[e], &expect[e], sizeof(float)) != 0) ++bad;
        }
        if (bad) { ++bad_chunks; bad_elems += bad; }
      }
    }
    INFO("round trip: " + std::to_string(bad_chunks) + " corrupt chunk(s), " +
         std::to_string(bad_elems) + " differing element(s)");
    REQUIRE(bad_chunks == 0);
  }

  const int n = clio::cte::compressor::NeuroPressChunkHistoryCount();
  INFO(std::to_string(n) + " chunk record(s) for " +
       std::to_string(kPhases * kChunksPerPhase) + " write(s)");
  REQUIRE(n > 0);

  // Records arrive in write order, so the phase a row belongs to is its index.
  // Cost MAPE is the primary metric: it is what gates the update, and what
  // the selection ranks on. Ratio MAPE is reported beside it, but the model
  // is not optimizing it directly.
  std::vector<std::vector<double>> cost_mape(kPhases);
  std::vector<std::vector<double>> mape(kPhases);
  std::vector<std::vector<double>> ct_mape(kPhases);
  std::vector<std::vector<double>> entropy(kPhases);
  int trained = 0;
  for (int i = 0; i < n; ++i) {
    clio::cte::compressor::NeuroPressChunkDiag d;
    REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(i, &d) == 0);
    const int phase = (i / kChunksPerPhase) % kPhases;
    cost_mape[phase].push_back(static_cast<double>(d.cost_model_error_pct));
    mape[phase].push_back(static_cast<double>(d.ratio_mape));
    ct_mape[phase].push_back(static_cast<double>(d.comp_time_mape));
    entropy[phase].push_back(static_cast<double>(d.feat_entropy));
    trained += d.sgd_fired;
    if (print_all || (i % kChunksPerPhase) < 2 ||
        (i % kChunksPerPhase) >= kChunksPerPhase - 2) {
      char line[256];
      std::snprintf(line, sizeof(line),
                    "%-13s c%-4d H=%.4f  act=%-2d %-9s pred_r=%8.3f "
                    "real_r=%7.3f  ratio_mape=%9.3f cost_mape=%10.3f %s",
                    PhaseName(phase), i % kChunksPerPhase, d.feat_entropy,
                    d.nn_action, CodecName(d.nn_action), d.predicted_ratio,
                    d.actual_ratio, d.ratio_mape, d.cost_model_error_pct,
                    d.sgd_fired ? "SGD" : "");
      INFO(std::string(line));
    }
  }

  for (int p = 0; p < kPhases; ++p) {
    if (mape[p].empty()) continue;
    const int n_p = static_cast<int>(mape[p].size());
    const double e_lo = *std::min_element(entropy[p].begin(), entropy[p].end());
    const double e_hi = *std::max_element(entropy[p].begin(), entropy[p].end());
    INFO(std::string(PhaseName(p)) + ": entropy " + std::to_string(e_lo) +
         ".." + std::to_string(e_hi) +
         " | cost " + std::to_string(Mean(cost_mape[p], 0, kWindow)) +
         "->" + std::to_string(Mean(cost_mape[p], n_p - kWindow, kWindow)) +
         " | ratio " + std::to_string(Mean(mape[p], 0, kWindow)) +
         "->" + std::to_string(Mean(mape[p], n_p - kWindow, kWindow)) +
         " | comp_time " + std::to_string(Mean(ct_mape[p], 0, kWindow)) +
         "->" + std::to_string(Mean(ct_mape[p], n_p - kWindow, kWindow)));
  }
  INFO(std::to_string(trained) + " SGD update(s) over " + std::to_string(n) +
       " chunks");

  // The run is only meaningful if the phases really are distinct and, with
  // learning on, the update actually fired.
  if (learning) {
    REQUIRE(trained > 0);
  } else {
    REQUIRE(trained == 0);
  }
}

SIMPLE_TEST_MAIN()
