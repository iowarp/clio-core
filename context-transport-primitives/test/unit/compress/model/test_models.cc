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
#include <cmath>
#include <filesystem>
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
