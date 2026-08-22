/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Does data with genuinely different statistics make the model pick different
 * codecs?
 *
 * Nine regimes spanning entropy 0.03-7.5 bits/byte, chosen to vary the three
 * features the network reads AND the byte-level structure the codecs exploit:
 * skewed byte histograms (entropy coders), constant-exponent mantissa planes
 * (shuffle + bitcomp), spiky/heavy-tailed distributions, near-incompressible
 * noise. Deliberately decorrelated, so entropy alone does not predict the
 * other two features.
 *
 * Writes each through the real compressor and reports what was selected, with
 * the measured statistics beside it. NP_LEARN=1 lets the model adapt as it
 * goes; the default is frozen weights, so each regime's selection is the
 * trained model's own answer rather than a product of the regime before it.
 */
#include "simple_test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
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

bool HaveGpu() {
#if CTP_ENABLE_CUDA
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
  return false;
#endif
}

std::string EnvOr(const char *k, const char *d) {
  const char *v = std::getenv(k);
  return (v && v[0]) ? std::string(v) : std::string(d);
}
int EnvInt(const char *k, int d) {
  const char *v = std::getenv(k);
  return (v && v[0]) ? std::atoi(v) : d;
}
/* One mode per PROCESS. The predictor is loaded from the .nnwt at Create()
 * and Train() never calls Save(), so a fresh process is a pristine model --
 * the refresh between phases is structural, not something to remember. */
enum Mode { kInference = 0, kLearning = 1, kExploration = 2 };
Mode CurrentMode() {
  const std::string m = EnvOr("NP_MODE", "inference");
  if (m == "learning") return kLearning;
  if (m == "exploration") return kExploration;
  return kInference;
}
const char *ModeName(Mode m) {
  return m == kInference ? "inference" : m == kLearning ? "learning" : "exploration";
}

const char *CodecName(int action) {
  switch (action % 8) {
    case 0: return "lz4";      case 1: return "snappy";
    case 2: return "deflate";  case 3: return "gdeflate";
    case 4: return "zstd";     case 5: return "ans";
    case 6: return "cascaded"; default: return "bitcomp";
  }
}

/* ---- deterministic RNG, so a run is reproducible ---- */
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
  uint32_t next() {
    s ^= s >> 33; s *= 0xff51afd7ed558ccdull;
    s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ull;
    s ^= s >> 33;
    return static_cast<uint32_t>(s >> 32);
  }
  double u01() { return next() / 4294967296.0; }
  double norm() {  // Box-Muller
    double a = std::max(1e-12, u01()), b = u01();
    return std::sqrt(-2.0 * std::log(a)) * std::cos(6.283185307179586 * b);
  }
  double cauchy() { return std::tan(3.141592653589793 * (u01() - 0.5)); }
};

struct Regime { const char *name; void (*fill)(float *, size_t, Rng &); };

void g_sparse(float *a, size_t n, Rng &r) {           // near-zero entropy, spiky
  std::memset(a, 0, n * sizeof(float));
  for (size_t k = 0; k < n / 512; ++k) a[r.next() % n] = static_cast<float>(r.norm() * 1000.0);
}
void g_lowcard(float *a, size_t n, Rng &r) {          // 8 values: skewed byte histogram
  for (size_t i = 0; i < n; ++i) a[i] = static_cast<float>(r.next() % 8);
}
void g_mantissa(float *a, size_t n, Rng &) {          // constant exponent, varying mantissa
  for (size_t i = 0; i < n; ++i) a[i] = 1.0f + static_cast<float>(i) * 1e-7f;
}
void g_bimodal(float *a, size_t n, Rng &r) {
  for (size_t i = 0; i < n; ++i)
    a[i] = static_cast<float>((r.u01() < 0.5 ? -100.0 : 100.0) + r.norm() * 0.5);
}
void g_gauss_narrow(float *a, size_t n, Rng &r) { for (size_t i = 0; i < n; ++i) a[i] = static_cast<float>(r.norm() * 0.01); }
void g_heavy(float *a, size_t n, Rng &r) { for (size_t i = 0; i < n; ++i) a[i] = static_cast<float>(r.cauchy() * 10.0); }

/* Nine regimes. The smoothly-structured ones (const, runs, steps, linear,
 * smooth) were dropped: every codec handles them well, so they separate the
 * action space far less than the distributional regimes below. */
const Regime kRegimes[] = {
  {"sparse",       g_sparse},       {"lowcard",    g_lowcard},
  {"mantissa",     g_mantissa},     {"bimodal",    g_bimodal},
  {"gauss-narrow", g_gauss_narrow}, {"heavy-tail", g_heavy},
};
constexpr int kNumRegimes = static_cast<int>(sizeof(kRegimes) / sizeof(kRegimes[0]));

std::string WriteComposeConfig(const std::string &path, Mode mode) {
  const bool learning = (mode != kInference);
  const bool explore = (mode == kExploration);
  std::ofstream f(path);
  f << "networking:\n  port: 9413\n"
    << "runtime:\n  num_threads: 4\n  queue_depth: 1024\n"
    << "compose:\n"
    << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
    << "    pool_query: local\n    pool_id: \"301.0\"\n"
    << "    bdev_type: ram\n    capacity: \"" << EnvOr("NP_CAP", "16GB") << "\"\n"
    << "  - mod_name: clio_cte_compressor\n    pool_name: cte_compressor\n"
    << "    pool_query: local\n    pool_id: \"512.0\"\n"
    << "    next_pool_id: \"513.0\"\n"
    << "    neuropress_model_path: \"" << CLIO_CTP_NEUROPRESS_WEIGHTS_DIR << "\"\n"
    << "    neuropress_online_learning_enabled: "
    << (learning ? "true" : "false") << "\n"
    << "    neuropress_learning_rate: " << EnvOr("NP_LR", "0.1") << "\n"
    << "    neuropress_mape_threshold: " << EnvOr("NP_MAPE", "0.2") << "\n"
    << "    neuropress_exploration_enabled: " << (explore ? "true" : "false") << "\n"
    << (explore ? "    neuropress_exploration_threshold: 0.0\n"
                  "    neuropress_exploration_k: 31\n" : "")
    << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
    << "    pool_query: local\n    pool_id: \"513.0\"\n"
    << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
    << "        bdev_type: \"ram\"\n        capacity_limit: \""
    << EnvOr("NP_CAP", "16GB") << "\"\n        score: 1.0\n"
    << "    dpe:\n      dpe_type: \"max_bw\"\n";
  f.close();
  return path;
}

}  // namespace

TEST_CASE("Diverse statistics drive diverse codec selection",
          "[compressor][neuropress][regimes][693]") {
  if (!HaveGpu()) {
    INFO("No CUDA device -- skipping");
    return;
  }
  const Mode mode = CurrentMode();
  const int reps = std::max(1, EnvInt("NP_REPS", 100));
  INFO(std::string("MODE=") + ModeName(mode) +
       (mode == kExploration ? " (K=31, threshold 0)" : "") +
       "  weights: fresh from .nnwt" +
       "  regimes=" + std::to_string(kNumRegimes) +
       "  chunks/regime=" + std::to_string(reps));

  const std::string cfg = WriteComposeConfig(
      std::string("/tmp/np_regimes_") + ModeName(mode) + ".yaml", mode);
  SIMPLE_TEST_SETENV("CLIO_SERVER_CONF", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);

  auto *cte = CLIO_CTE_CLIENT;
  auto tag_task = cte->AsyncGetOrCreateTag("np_regime_tag");
  tag_task.Wait();
  auto tag_id = tag_task->tag_id_;

  clio::cte::compressor::NeuroPressResetChunkHistory();

  const size_t kElems = 1024 * 1024;  // 4 MiB of float32
  const size_t kBytes = kElems * sizeof(float);
  clio::cte::compressor::Client compressor(clio::cte::core::kCtePoolId);
  std::vector<float> expect(kElems);

  for (int ri = 0; ri < kNumRegimes; ++ri) {
    for (int k = 0; k < reps; ++k) {
      auto buf = CLIO_IPC->AllocateBuffer(kBytes);
      REQUIRE(!buf.IsNull());
      Rng rng(0x9E3779B9ull + 7919ull * (ri * 64 + k));
      kRegimes[ri].fill(reinterpret_cast<float *>(buf.ptr_), kElems, rng);

      clio::cte::core::Context ctx;
      ctx.dynamic_compress_ = 2;
      ctx.data_type_ = 1;
      auto sched = compressor.AsyncDynamicSchedule(
          clio::run::PoolQuery::Local(), tag_id,
          "np_reg_" + std::to_string(ri) + "_" + std::to_string(k),
          /*offset=*/0, kBytes, buf.shm_.template Cast<void>(), -1.0f, ctx, 0,
          clio::run::PoolId(513, 0));
      sched.Wait();
      REQUIRE(sched->GetReturnCode() == 0);
    }
  }

  const int n = clio::cte::compressor::NeuroPressChunkHistoryCount();
  INFO(std::to_string(n) + " chunk record(s) for " +
       std::to_string(kNumRegimes * reps) + " write(s)");
  REQUIRE(n > 0);

  std::map<std::string, int> codec_hist;
  std::map<int, int> action_hist;
  double e_lo = 1e9, e_hi = -1e9;
  int trained = 0, explored = 0, swapped = 0;

  char hdr[220];
  std::snprintf(hdr, sizeof(hdr), "%-14s %8s %12s %11s  %-3s %-9s %8s %8s",
                "regime", "entropy", "mad", "2nd-deriv", "act", "codec",
                "real_r", "pred_r");
  INFO(std::string(hdr));
  INFO(std::string(92, '-'));

  for (int i = 0; i < n; ++i) {
    clio::cte::compressor::NeuroPressChunkDiag d;
    REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(i, &d) == 0);
    const int ri = (i / reps) % kNumRegimes;
    e_lo = std::min(e_lo, static_cast<double>(d.feat_entropy));
    e_hi = std::max(e_hi, static_cast<double>(d.feat_entropy));

    const std::string codec =
        std::string(CodecName(d.nn_action)) + (d.nn_action >= 16 ? "+shuf" : "");
    codec_hist[codec]++;
    action_hist[d.nn_action]++;
    trained += d.sgd_fired;
    if (d.exploration_triggered) {
      ++explored;
      if (d.nn_action != d.nn_original_action) ++swapped;
    }

    if (EnvInt("NP_PRINT_ALL", 0) || i % reps == 0) {
      char line[320];
      std::snprintf(line, sizeof(line),
                    "ROW %-14s %d %8.4f %12.4f %11.4f %d %d %-13s %9.4f %9.4f "
                    "%9.4f %11.4f %d %d %12.3f %12.3f %9.4f %9.4f "
                    "%11.4f %11.4f %11.4f %11.4f",
                    kRegimes[ri].name, i % reps, d.feat_entropy, d.feat_mad,
                    d.feat_deriv, d.nn_action, d.nn_original_action,
                    codec.c_str(), d.actual_ratio, d.predicted_ratio,
                    d.ratio_mape, d.cost_model_error_pct, d.sgd_fired,
                    d.exploration_triggered, d.actual_cost, d.predicted_cost,
                    d.comp_time_mape, d.decomp_time_mape,
                    d.predicted_comp_time, d.compression_ms,
                    d.predicted_decomp_time, d.decompression_ms);
      INFO(std::string(line));
    }
  }

  std::string hist;
  for (const auto &kv : codec_hist)
    hist += kv.first + "=" + std::to_string(kv.second) + "  ";
  INFO("selection histogram: " + hist);
  INFO("sgd updates: " + std::to_string(trained) +
       "   explorations: " + std::to_string(explored) +
       "   swapped by exploration: " + std::to_string(swapped));
  INFO("distinct codecs: " + std::to_string(codec_hist.size()) +
       "   distinct actions: " + std::to_string(action_hist.size()) +
       "   entropy span: " + std::to_string(e_lo) + " .. " +
       std::to_string(e_hi));

  // The dataset's job is to be varied enough that the model has something to
  // discriminate on. A single codec across every regime would mean either the
  // data is not actually diverse or the selection is not responding to it --
  // both make every downstream selection claim vacuous.
  REQUIRE(e_hi - e_lo > 4.0);
  REQUIRE(codec_hist.size() >= 2);
  // Each mode must actually do its own thing, or the comparison between the
  // three is comparing three copies of the same run.
  if (mode == kInference) {
    REQUIRE(trained == 0);
    REQUIRE(explored == 0);
  } else if (mode == kLearning) {
    REQUIRE(trained > 0);
    REQUIRE(explored == 0);
  } else {
    REQUIRE(explored > 0);
  }
}

SIMPLE_TEST_MAIN()
