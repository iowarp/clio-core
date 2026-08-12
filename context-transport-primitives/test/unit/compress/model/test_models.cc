/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_models.cc
 * @brief Unit tests for the unified compression-metric predictors (issue #693).
 *
 * Exercises the shared CompressionPredictor interface across all four model
 * backends: Q-table and linreg-table (pure C++, trained inline then round-tripped
 * through Save/Load), the NeuroPress NN (not-ready safety + missing-model load),
 * and the XGBoost stub (disabled by default, must degrade gracefully).
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/model/linreg_table_predictor.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/qtable_predictor.h"
#include "clio_ctp/compress/model/ranking.h"
#include "clio_ctp/compress/model/xgboost_predictor.h"

using namespace ctp::compress::model;  // NOLINT(build/namespaces)

// Test-harness hooks required by test/unit/main.cc (no global setup needed).
void MainPretest() {}
void MainPosttest() {}

namespace {

// Synthetic training set where compression ratio is anti-correlated with
// entropy (low-entropy data compresses better) — enough structure for the
// table models to learn a non-trivial mapping.
std::vector<CompressionFeatures> MakeFeatures() {
  std::vector<CompressionFeatures> feats;
  for (int lib = 1; lib <= 3; ++lib) {
    for (int e = 1; e <= 8; ++e) {
      CompressionFeatures f;
      f.chunk_size_bytes = 1024.0 * e;
      f.target_cpu_util = 50.0;
      f.shannon_entropy = static_cast<double>(e);
      f.mad = 0.1 * e;
      f.second_derivative_mean = 0.01 * e;
      f.library_config_id = lib * 10 + 2;  // BALANCED preset
      f.config_balanced = 1.0;
      f.data_type_float = 1.0;
      feats.push_back(f);
    }
  }
  return feats;
}

std::vector<TrainingLabels> MakeLabels(
    const std::vector<CompressionFeatures> &feats) {
  std::vector<TrainingLabels> labels;
  labels.reserve(feats.size());
  for (const auto &f : feats) {
    float ratio = static_cast<float>(9.0 - f.shannon_entropy);  // 1..8
    float psnr = 0.0f;
    float time = static_cast<float>(f.chunk_size_bytes / 1.0e6);
    labels.emplace_back(ratio, psnr, time);
  }
  return labels;
}

std::string TempDir(const std::string &name) {
  std::filesystem::path base =
      std::filesystem::temp_directory_path() / "ctp_compress_model_test";
  std::filesystem::create_directories(base / name);
  return (base / name).string();
}

}  // namespace

TEST_CASE("QTablePredictor trains, predicts, and round-trips") {
  auto feats = MakeFeatures();
  auto labels = MakeLabels(feats);

  QTablePredictor q;
  REQUIRE(q.Type() == ModelType::kQTable);
  REQUIRE(q.Train(feats, labels));
  REQUIRE(q.IsReady());

  auto pred = q.Predict(feats.front());
  REQUIRE(std::isfinite(pred.compression_ratio));
  REQUIRE(pred.compression_ratio > 0.0);

  std::string dir = TempDir("qtable");
  REQUIRE(q.Save(dir));
  QTablePredictor q2;
  REQUIRE(q2.Load(dir));
  REQUIRE(q2.IsReady());
  auto pred2 = q2.Predict(feats.front());
  REQUIRE(std::isfinite(pred2.compression_ratio));
}

TEST_CASE("LinRegTablePredictor trains, predicts, and round-trips") {
  auto feats = MakeFeatures();
  auto labels = MakeLabels(feats);

  LinRegTablePredictor lr;
  REQUIRE(lr.Type() == ModelType::kLinRegTable);
  REQUIRE(lr.Train(feats, labels));
  REQUIRE(lr.IsReady());

  auto pred = lr.Predict(feats.back());
  REQUIRE(std::isfinite(pred.compression_ratio));

  std::string dir = TempDir("linreg");
  REQUIRE(lr.Save(dir));
  LinRegTablePredictor lr2;
  REQUIRE(lr2.Load(dir));
  REQUIRE(lr2.IsReady());
  auto pred2 = lr2.Predict(feats.back());
  REQUIRE(std::isfinite(pred2.compression_ratio));
}

TEST_CASE("NeuroPressNNPredictor is safe before a model is loaded") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Type() == ModelType::kNeuroPressNN);
  REQUIRE_FALSE(nn.IsReady());

  // Predict before load must not crash and must return finite values.
  CompressionFeatures f;
  f.chunk_size_bytes = 4096;
  f.shannon_entropy = 4.0;
  auto pred = nn.Predict(f);
  REQUIRE(std::isfinite(pred.compression_ratio));

  // Loading a non-existent model directory fails cleanly.
  REQUIRE_FALSE(nn.Load(TempDir("nn_missing") + "/does_not_exist"));
  REQUIRE_FALSE(nn.IsReady());
}

TEST_CASE("Rank() bulk-ranks candidates best-first (primary API)") {
  auto feats = MakeFeatures();
  auto labels = MakeLabels(feats);
  QTablePredictor q;
  REQUIRE(q.Train(feats, labels));

  DataFeatures data;
  data.chunk_size_bytes = 4096;
  data.shannon_entropy = 3.0;
  data.mad = 0.3;
  data.second_derivative_mean = 0.03;
  data.data_type_float = 1.0;

  // CPU-only default candidate set: every known CPU compressor x 3 presets.
  auto candidates = DefaultCandidates(/*include_gpu=*/false);
  REQUIRE(candidates.size() > 1);

  auto ranked = q.Rank(data, candidates);
  REQUIRE(ranked.size() == candidates.size());
  // Results must be sorted best-first and carry finite predictions.
  for (size_t i = 0; i < ranked.size(); ++i) {
    REQUIRE(std::isfinite(ranked[i].prediction.compression_ratio));
    REQUIRE(std::isfinite(ranked[i].score));
    if (i > 0) REQUIRE(ranked[i - 1].score >= ranked[i].score);
  }

  // Weighting by compression time must be able to reorder the ranking away
  // from the ratio-only default (different objective -> different winner set).
  RankingWeights time_first;
  time_first.w_ratio = 0.0;
  time_first.w_compress_time = 1.0;  // Score = -time: prefer the fastest.
  auto ranked_time = q.Rank(data, candidates, time_first);
  REQUIRE(ranked_time.size() == candidates.size());
  for (size_t i = 1; i < ranked_time.size(); ++i) {
    REQUIRE(ranked_time[i - 1].score >= ranked_time[i].score);
  }
}

TEST_CASE("DefaultCandidates(gpu) covers NeuroPress's full 8-algorithm "
          "action space") {
  // NeuroPress's NN action space is 8 GPU algorithms (LZ4, Snappy, Deflate,
  // Gdeflate, Zstd, ANS, Cascaded, Bitcomp) x 2 preprocessing options. Six of
  // the eight are already registered; Cascaded and Bitcomp are the gap.
  const std::vector<std::string> kNeuroPressAlgos = {
      "nvcomp-lz4",  "nvcomp-snappy",   "nvcomp-deflate", "nvcomp-gdeflate",
      "nvcomp-zstd", "nvcomp-ans",      "nvcomp-cascaded", "nvcomp-bitcomp"};
  for (const auto &algo : kNeuroPressAlgos) {
    bool found = false;
    for (const auto &e : KnownCompressors()) {
      if (algo == e.name) {
        found = true;
        REQUIRE(e.is_gpu);
        break;
      }
    }
    INFO("missing NeuroPress action-space algorithm: " << algo);
    REQUIRE(found);
  }

  // Every entry must carry a unique, non-zero base_id (frozen ML-scheme id).
  std::vector<int> ids;
  for (const auto &e : KnownCompressors()) {
    REQUIRE(e.base_id != 0);
    ids.push_back(e.base_id);
  }
  std::sort(ids.begin(), ids.end());
  REQUIRE(std::adjacent_find(ids.begin(), ids.end()) == ids.end());

  auto gpu_candidates = DefaultCandidates(/*include_gpu=*/true);
  int cascaded_count = 0, bitcomp_count = 0;
  for (const auto &c : gpu_candidates) {
    if (c.library_name == "nvcomp-cascaded") ++cascaded_count;
    if (c.library_name == "nvcomp-bitcomp") ++bitcomp_count;
  }
  REQUIRE(cascaded_count > 0);
  REQUIRE(bitcomp_count > 0);
}

TEST_CASE("DefaultCandidates(include_cpu=false) matches the original "
          "NeuroPress project's GPU-only action space") {
  // gpucompress_compress.cpp (the original NeuroPress project): "Stats
  // remain on GPU -- NN inference reads d_stats_ptr directly on device."
  // Its CompressionAlgorithm enum has zero CPU entries. A caller resolving
  // a device-resident buffer must be able to reproduce that: GPU
  // candidates present, CPU candidates entirely absent (not just
  // deprioritized), so nothing forces a host read of the buffer.
  auto gpu_only = DefaultCandidates(/*include_gpu=*/true, {1, 2, 3},
                                     /*with_preprocessors=*/false,
                                     /*error_bound=*/1e-3,
                                     /*include_cpu=*/false);
  REQUIRE(!gpu_only.empty());
  for (const auto &c : gpu_only) {
    const CompressorEntry *entry = nullptr;
    for (const auto &e : KnownCompressors()) {
      if (e.base_id == c.base_id) { entry = &e; break; }
    }
    REQUIRE(entry != nullptr);
    INFO("unexpected CPU candidate in GPU-only set: " << entry->name);
    REQUIRE(entry->is_gpu);
  }

  // Same count as the full GPU set (include_cpu only removes CPU rows).
  auto gpu_and_cpu = DefaultCandidates(/*include_gpu=*/true);
  size_t gpu_row_count = 0;
  for (const auto &e : KnownCompressors()) {
    if (e.is_gpu) ++gpu_row_count;
  }
  REQUIRE(gpu_only.size() == gpu_row_count * 3);  // x3 presets
  REQUIRE(gpu_only.size() < gpu_and_cpu.size());

  // include_cpu=false with include_gpu=false (the default) leaves nothing
  // to rank -- callers must pass include_gpu=true alongside it.
  auto empty = DefaultCandidates(/*include_gpu=*/false, {1, 2, 3}, false,
                                  1e-3, /*include_cpu=*/false);
  REQUIRE(empty.empty());
}

#ifdef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
TEST_CASE("NeuroPressNNPredictor loads the real pretrained weights and "
          "ranks GPU candidates sensibly") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  // Highly compressible: near-zero entropy/MAD/curvature (e.g. a flat or
  // slowly-varying field).
  DataFeatures compressible;
  compressible.chunk_size_bytes = 4 * 1024 * 1024;
  compressible.shannon_entropy = 0.5;
  compressible.mad = 0.02;
  compressible.second_derivative_mean = 0.01;
  compressible.data_type_float = 1.0;

  // Near-incompressible: high entropy/MAD (e.g. noise).
  DataFeatures noisy = compressible;
  noisy.shannon_entropy = 7.8;
  noisy.mad = 0.9;
  noisy.second_derivative_mean = 0.8;

  auto gpu_candidates = DefaultCandidates(/*include_gpu=*/true);
  REQUIRE(gpu_candidates.size() > DefaultCandidates(false).size());

  auto ranked_compressible = RankDefault(nn, compressible, RankingWeights(),
                                         /*include_gpu=*/true);
  auto ranked_noisy = RankDefault(nn, noisy, RankingWeights(),
                                  /*include_gpu=*/true);
  REQUIRE(ranked_compressible.size() == gpu_candidates.size());
  REQUIRE(ranked_noisy.size() == gpu_candidates.size());

  // Sorted best-first, finite predictions, on both inputs.
  for (const auto *ranked : {&ranked_compressible, &ranked_noisy}) {
    for (size_t i = 0; i < ranked->size(); ++i) {
      REQUIRE(std::isfinite((*ranked)[i].prediction.compression_ratio));
      REQUIRE(std::isfinite((*ranked)[i].score));
      if (i > 0) REQUIRE((*ranked)[i - 1].score >= (*ranked)[i].score);
    }
  }

  // Behavioral sanity: the model must actually respond to the data
  // statistics, not just return a constant. The winning candidate's
  // predicted ratio for near-zero-entropy data should beat the winning
  // candidate's predicted ratio for high-entropy noise.
  REQUIRE(ranked_compressible.front().prediction.compression_ratio >
          ranked_noisy.front().prediction.compression_ratio);

  // The GPU candidate set (nvcomp/cusz/ndzip) must actually be reachable by
  // ranking, not just present in DefaultCandidates() -- confirm at least one
  // GPU compressor (base_id >= 13 per the CompressionFactory registry)
  // appears somewhere in the ranked results.
  bool saw_gpu_candidate = false;
  for (const auto &r : ranked_compressible) {
    if (r.candidate.base_id >= 13) {
      saw_gpu_candidate = true;
      break;
    }
  }
  REQUIRE(saw_gpu_candidate);
}

TEST_CASE("NeuroPressNNPredictor inverse-transforms log1p-encoded outputs "
          "with expm1, matching NeuroPress's own training/export") {
  // NeuroPress trains comp_time/decomp_time/ratio in log1p() space
  // (neural_net/core/data.py's OUTPUT_INVERSE) and its own CUDA inference
  // applies expm1f to undo it (src/nn/nn_gpu.cu). A plain linear inverse
  // transform silently leaves the log-space value in place. Empirically,
  // the raw (pre-fix) predicted ratio for this exact input/candidate set
  // never exceeds ~5.2 (log1p-space is naturally bounded for realistic
  // trained data) -- a ratio above 10 is only reachable once expm1 is
  // applied and the value is in real units.
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  DataFeatures compressible;
  compressible.chunk_size_bytes = 4 * 1024 * 1024;
  compressible.shannon_entropy = 0.5;
  compressible.mad = 0.02;
  compressible.second_derivative_mean = 0.01;
  compressible.data_type_float = 1.0;

  auto ranked = RankDefault(nn, compressible, RankingWeights(),
                            /*include_gpu=*/true);
  REQUIRE(!ranked.empty());

  bool saw_expm1_scale_ratio = false;
  for (const auto &r : ranked) {
    if (r.prediction.compression_ratio > 10.0) {
      saw_expm1_scale_ratio = true;
      break;
    }
  }
  REQUIRE(saw_expm1_scale_ratio);
}

TEST_CASE("NeuroPressNNPredictor's algo_id encoding does not collide "
          "distinct nvcomp algorithms") {
  // Regression test for issue #693: algo_id was computed as base_id % 8,
  // which collides nvcomp-zstd (base_id 15) with nvcomp-cascaded (23), and
  // nvcomp-gdeflate (16) with nvcomp-bitcomp (24) -- both pairs share a
  // remainder mod 8 despite being distinct one-hot algorithm indices in
  // NeuroPress's own trained action space (ALGORITHM_NAMES in
  // neural_net/core/configs.py: lz4=0, snappy=1, deflate=2, gdeflate=3,
  // zstd=4, ans=5, cascaded=6, bitcomp=7). Confirmed bit-identical
  // predictions for both pairs before the fix.
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  DataFeatures data;
  data.chunk_size_bytes = 1024 * 1024;
  data.shannon_entropy = 4.0;
  data.mad = 0.3;
  data.second_derivative_mean = 0.2;
  data.data_type_float = 1.0;

  auto predict_for = [&](int base_id) {
    CandidateConfig c;
    c.base_id = base_id;
    c.preset_id = 2;
    return nn.Predict(MakeCompressionFeatures(data, c));
  };

  auto zstd = predict_for(15);
  auto cascaded = predict_for(23);
  auto gdeflate = predict_for(16);
  auto bitcomp = predict_for(24);

  REQUIRE(zstd.compression_ratio != cascaded.compression_ratio);
  REQUIRE(gdeflate.compression_ratio != bitcomp.compression_ratio);
}

TEST_CASE("NeuroPressNNPredictor::Train() moves predictions toward "
          "real observed outcomes (issue #693 Cycle 4)") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  DataFeatures data;
  data.chunk_size_bytes = 1024 * 1024;
  data.shannon_entropy = 3.0;
  data.mad = 0.2;
  data.second_derivative_mean = 0.15;
  data.data_type_float = 1.0;

  CandidateConfig candidate;
  candidate.base_id = 15;  // nvcomp-zstd
  candidate.preset_id = 2;
  auto features = MakeCompressionFeatures(data, candidate);

  double before = nn.Predict(features).compression_ratio;

  // A real, consistent, observed outcome for this exact candidate --
  // repeated Train() calls (mirrors how compressor_runtime.cc will feed
  // one real (predicted, actual) pair per real compression) should pull
  // the prediction toward it, not leave it unchanged or move it away.
  const float kTrueRatio = 20.0f;
  TrainingLabels label(kTrueRatio, /*psnr=*/0.0f, /*comp_time_ms=*/5.0f,
                       /*decomp_time_ms=*/2.0f);
  std::vector<CompressionFeatures> batch_feats(1, features);
  std::vector<TrainingLabels> batch_labels(1, label);

  bool any_update_applied = false;
  for (int i = 0; i < 200; ++i) {
    if (nn.Train(batch_feats, batch_labels)) any_update_applied = true;
  }
  REQUIRE(any_update_applied);

  double after = nn.Predict(features).compression_ratio;
  REQUIRE(std::isfinite(after));
  INFO("before=" << before << " after=" << after << " true=" << kTrueRatio);
  REQUIRE(std::fabs(after - kTrueRatio) < std::fabs(before - kTrueRatio));
}

TEST_CASE("NeuroPressNNPredictor::Train() stays numerically stable under "
          "repeated adversarial/out-of-distribution labels") {
  // Regression coverage for the clipping/clamping/trust-region machinery
  // (gradient clipping, weight clamp, NaN/Inf sanitization) -- without it,
  // extreme labels cause weight explosion or NaN propagation within a few
  // SGD steps.
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  DataFeatures data;
  data.chunk_size_bytes = 512;
  data.shannon_entropy = 7.9;
  data.mad = 0.95;
  data.second_derivative_mean = 0.9;
  data.data_type_float = 0.0;

  std::vector<CompressionFeatures> batch_feats;
  std::vector<TrainingLabels> batch_labels;
  for (int base_id : {13, 15, 18, 23}) {
    CandidateConfig c;
    c.base_id = base_id;
    c.preset_id = 3;
    batch_feats.push_back(MakeCompressionFeatures(data, c));
    // Deliberately extreme/conflicting: huge ratio, tiny comp_time, huge
    // decomp_time, near-zero PSNR -- pushes every output head hard, and
    // some pairs are directionally conflicting for the shared layers.
    batch_labels.emplace_back(9999.0f, 0.01f, 0.02f, 4999.0f);
  }

  for (int i = 0; i < 500; ++i) {
    nn.Train(batch_feats, batch_labels);
  }

  for (const auto &f : batch_feats) {
    auto pred = nn.Predict(f);
    REQUIRE(std::isfinite(pred.compression_ratio));
    REQUIRE(std::isfinite(pred.compression_time_ms));
    REQUIRE(std::isfinite(pred.psnr_db));
  }
}

TEST_CASE("NeuroPressNNPredictor::Train() input validation") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));

  std::vector<CompressionFeatures> feats(2);
  std::vector<TrainingLabels> labels(3);  // mismatched size
  REQUIRE_FALSE(nn.Train(feats, labels));
  REQUIRE_FALSE(nn.Train({}, {}));
}
#endif  // CLIO_CTP_NEUROPRESS_WEIGHTS_DIR

TEST_CASE("NeuroPressNNPredictor::Train() is safe on a not-ready model") {
  NeuroPressNNPredictor nn;  // never loaded
  std::vector<CompressionFeatures> feats(1);
  std::vector<TrainingLabels> labels(1);
  REQUIRE_FALSE(nn.Train(feats, labels));
}

TEST_CASE("Rank() is safe on a not-ready model") {
  NeuroPressNNPredictor nn;  // never loaded
  DataFeatures data;
  data.chunk_size_bytes = 2048;
  data.shannon_entropy = 5.0;
  auto ranked = nn.Rank(data, DefaultCandidates(false));
  REQUIRE(ranked.size() == DefaultCandidates(false).size());
  for (const auto &r : ranked) {
    REQUIRE(std::isfinite(r.prediction.compression_ratio));
  }
}

TEST_CASE("XGBoostPredictor degrades gracefully when disabled") {
  XGBoostPredictor xgb;
  REQUIRE(xgb.Type() == ModelType::kXGBoost);
  // Default build has CTP_ENABLE_XGBOOST=0: the model is a not-ready stub.
  if (!xgb.IsReady()) {
    CompressionFeatures f;
    f.chunk_size_bytes = 2048;
    auto pred = xgb.Predict(f);
    REQUIRE(std::isfinite(pred.compression_ratio));
  }
}

#ifdef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
TEST_CASE("TrainDecompHead updates ONLY the decompression-time head") {
  // Port of NeuroPress's nnBatchedDecompSGDKernel (nn_gpu.cu), which forwards
  // through the frozen trunk and writes only w5 row 1 / b5[1]. A
  // decompression-time miss must never perturb the shared representation the
  // other three heads depend on.
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  CompressionFeatures f;
  f.chunk_size_bytes = 2 * 1024 * 1024;
  f.shannon_entropy = 3.0;
  f.mad = 0.3;
  f.second_derivative_mean = 0.03;
  f.data_type_float = 1.0;
  f.library_config_id = 13 * 10 + 2;  // nvcomp-lz4, BALANCED
  f.config_balanced = 1.0;

  // Baseline predictions for a DIFFERENT candidate, to prove the trunk and
  // the other heads are untouched.
  CompressionFeatures other = f;
  other.library_config_id = 18 * 10 + 2;  // nvcomp-ans
  auto before_other = nn.Predict(other);
  auto before = nn.Predict(f);

  // A decompression time far from whatever the model currently predicts, so
  // the error clears the 0.05 noise gate.
  double absurd_ms = (before.decompression_time_ms > 50.0) ? 1.0 : 900.0;
  REQUIRE(nn.TrainDecompHead({f}, {absurd_ms}));

  auto after = nn.Predict(f);
  auto after_other = nn.Predict(other);

  // The decomp head moved, and moved TOWARD the observed value.
  REQUIRE(after.decompression_time_ms != before.decompression_time_ms);
  REQUIRE(std::fabs(after.decompression_time_ms - absurd_ms) <
          std::fabs(before.decompression_time_ms - absurd_ms));

  // The other three heads are byte-identical: a shared-trunk update would
  // have perturbed all of them.
  REQUIRE(after.compression_ratio == before.compression_ratio);
  REQUIRE(after.compression_time_ms == before.compression_time_ms);
  REQUIRE(after.psnr_db == before.psnr_db);
  REQUIRE(after_other.compression_ratio == before_other.compression_ratio);
  REQUIRE(after_other.compression_time_ms == before_other.compression_time_ms);

  // Guards: mismatched lengths, and a non-measured time, are no-ops.
  REQUIRE_FALSE(nn.TrainDecompHead({f}, {}));
  REQUIRE_FALSE(nn.TrainDecompHead({f}, {0.0}));
  REQUIRE_FALSE(nn.TrainDecompHead({f}, {-1.0}));
}

TEST_CASE("Train() leaves the decompression-time head to TrainDecompHead") {
  // Upstream's nnSGDKernel skips output 1 (`if (out == 1) continue;` / `t != 1`)
  // because the deferred pass owns that head. Checked on the WEIGHTS, not on
  // the prediction: Train() updates the shared trunk (W1-W4), which shifts
  // every output including output 1 -- upstream behaves the same way. The
  // invariant is specifically that w5 row 1 and b5[1] are untouched.
  //
  // .nnwt layout (see neuropress_nn_predictor.h): 24-byte header, 32 norm
  // floats, then per-layer [weights, biases] with layer 4 = 64x8.
  constexpr size_t kHeaderBytes = 24;
  constexpr size_t kNormFloats = 32;
  constexpr size_t kL0 = 8 * 64 + 64;
  constexpr size_t kLMid = 64 * 64 + 64;
  constexpr size_t kW5Base = kL0 + 3 * kLMid;      // start of layer-4 weights
  constexpr size_t kW5Row1 = kW5Base + 1 * 64;     // 64 floats
  constexpr size_t kB5Base = kW5Base + 64 * 8;
  constexpr size_t kB5Idx1 = kB5Base + 1;

  auto read_floats = [](const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    std::vector<float> out;
    if (raw.size() <= kHeaderBytes) return out;
    size_t n = (raw.size() - kHeaderBytes) / sizeof(float);
    out.resize(n);
    std::memcpy(out.data(), raw.data() + kHeaderBytes, n * sizeof(float));
    return out;
  };

  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  const std::string dir_a = "/tmp/clio_np_head_before";
  const std::string dir_b = "/tmp/clio_np_head_after";
  std::filesystem::create_directories(dir_a);
  std::filesystem::create_directories(dir_b);
  REQUIRE(nn.Save(dir_a));

  CompressionFeatures f;
  f.chunk_size_bytes = 1024 * 1024;
  f.shannon_entropy = 6.0;
  f.mad = 0.5;
  f.second_derivative_mean = 0.4;
  f.data_type_float = 1.0;
  f.library_config_id = 15 * 10 + 2;  // nvcomp-zstd, BALANCED
  f.config_balanced = 1.0;

  // Supply a real decompression-time label; Train() must still not write that
  // head's weights.
  std::vector<TrainingLabels> labels = {
      TrainingLabels(/*ratio=*/9.0f, /*psnr=*/0.0f, /*time=*/40.0f,
                     /*decompress_time=*/777.0f)};
  REQUIRE(nn.Train({f}, labels));
  REQUIRE(nn.Save(dir_b));

  auto before = read_floats(dir_a + "/model.nnwt");
  auto after = read_floats(dir_b + "/model.nnwt");
  REQUIRE(before.size() == after.size());
  REQUIRE(after.size() > kNormFloats + kB5Idx1);

  // The decomp head is byte-identical.
  for (size_t i = 0; i < 64; ++i) {
    INFO("w5 row 1 element " << i << " was modified by Train()");
    REQUIRE(after[kNormFloats + kW5Row1 + i] ==
            before[kNormFloats + kW5Row1 + i]);
  }
  REQUIRE(after[kNormFloats + kB5Idx1] == before[kNormFloats + kB5Idx1]);

  // Sanity: Train() did update other weights, so the check above is not
  // passing merely because nothing happened.
  bool saw_change = false;
  for (size_t i = 0; i < before.size(); ++i) {
    if (after[i] != before[i]) { saw_change = true; break; }
  }
  REQUIRE(saw_change);
}

#endif  // CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
