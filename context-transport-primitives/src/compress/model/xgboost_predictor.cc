/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file xgboost_predictor.cc
 * @brief XGBoost-based compression-metric predictor implementation.
 */

#include "clio_ctp/compress/model/xgboost_predictor.h"
#include <chrono>
#include <fstream>
#include <sstream>

#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
#include <filesystem>
namespace fs = std::filesystem;
#define XGBOOST_CHECK(call)                                                   \
  do {                                                                         \
    int ret = (call);                                                         \
    if (ret != 0) return false;                                               \
  } while (0)
#endif

namespace ctp::compress::model {

XGBoostPredictor::XGBoostPredictor() = default;

XGBoostPredictor::~XGBoostPredictor() {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  std::lock_guard<std::mutex> lock(mutex_);
  if (ratio_booster_ != nullptr) {
    XGBoosterFree(ratio_booster_);
  }
  if (psnr_booster_ != nullptr) {
    XGBoosterFree(psnr_booster_);
  }
  if (time_booster_ != nullptr) {
    XGBoosterFree(time_booster_);
  }
#endif
}

XGBoostPredictor::XGBoostPredictor(XGBoostPredictor&& other) noexcept {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  ratio_booster_ = other.ratio_booster_;
  psnr_booster_ = other.psnr_booster_;
  time_booster_ = other.time_booster_;
  is_ready_ = other.is_ready_;
  other.ratio_booster_ = nullptr;
  other.psnr_booster_ = nullptr;
  other.time_booster_ = nullptr;
  other.is_ready_ = false;
#endif
}

XGBoostPredictor& XGBoostPredictor::operator=(
    XGBoostPredictor&& other) noexcept {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  if (this != &other) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ratio_booster_ != nullptr) {
      XGBoosterFree(ratio_booster_);
    }
    if (psnr_booster_ != nullptr) {
      XGBoosterFree(psnr_booster_);
    }
    if (time_booster_ != nullptr) {
      XGBoosterFree(time_booster_);
    }
    ratio_booster_ = other.ratio_booster_;
    psnr_booster_ = other.psnr_booster_;
    time_booster_ = other.time_booster_;
    is_ready_ = other.is_ready_;
    other.ratio_booster_ = nullptr;
    other.psnr_booster_ = nullptr;
    other.time_booster_ = nullptr;
    other.is_ready_ = false;
  }
#endif
  return *this;
}

bool XGBoostPredictor::Load(const std::string& model_dir) {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  std::lock_guard<std::mutex> lock(mutex_);
  if (ratio_booster_ != nullptr) {
    XGBoosterFree(ratio_booster_);
    ratio_booster_ = nullptr;
  }
  if (psnr_booster_ != nullptr) {
    XGBoosterFree(psnr_booster_);
    psnr_booster_ = nullptr;
  }
  if (time_booster_ != nullptr) {
    XGBoosterFree(time_booster_);
    time_booster_ = nullptr;
  }
  is_ready_ = false;

  fs::path dir(model_dir);
  fs::path ratio_path = dir / "compression_ratio_model.json";
  fs::path psnr_path = dir / "psnr_model.json";
  fs::path time_path = dir / "compression_time_model.json";

  if (!fs::exists(ratio_path)) return false;

  XGBOOST_CHECK(XGBoosterCreate(nullptr, 0, &ratio_booster_));
  XGBOOST_CHECK(
      XGBoosterLoadModel(ratio_booster_, ratio_path.string().c_str()));

  if (fs::exists(psnr_path)) {
    XGBOOST_CHECK(XGBoosterCreate(nullptr, 0, &psnr_booster_));
    XGBOOST_CHECK(XGBoosterLoadModel(psnr_booster_, psnr_path.string().c_str()));
  }

  if (fs::exists(time_path)) {
    XGBOOST_CHECK(XGBoosterCreate(nullptr, 0, &time_booster_));
    XGBOOST_CHECK(XGBoosterLoadModel(time_booster_, time_path.string().c_str()));
  }

  is_ready_ = true;
  return true;
#else
  return false;
#endif
}

bool XGBoostPredictor::Save(const std::string& model_dir) {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  std::lock_guard<std::mutex> lock(mutex_);
  if (!is_ready_ || ratio_booster_ == nullptr) {
    return false;
  }

  fs::path dir(model_dir);
  if (!fs::exists(dir)) {
    fs::create_directories(dir);
  }

  fs::path ratio_path = dir / "compression_ratio_model.json";
  XGBOOST_CHECK(XGBoosterSaveModel(ratio_booster_, ratio_path.string().c_str()));

  if (psnr_booster_ != nullptr) {
    fs::path psnr_path = dir / "psnr_model.json";
    XGBOOST_CHECK(XGBoosterSaveModel(psnr_booster_, psnr_path.string().c_str()));
  }

  if (time_booster_ != nullptr) {
    fs::path time_path = dir / "compression_time_model.json";
    XGBOOST_CHECK(XGBoosterSaveModel(time_booster_, time_path.string().c_str()));
  }

  return true;
#else
  return false;
#endif
}

bool XGBoostPredictor::IsReady() const {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  return is_ready_;
#else
  return false;
#endif
}

CompressionPrediction XGBoostPredictor::Predict(
    const CompressionFeatures& features) {
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  std::vector<CompressionFeatures> batch = {features};
  auto results = PredictBatch(batch);
  if (!results.empty()) {
    return results[0];
  }
  return CompressionPrediction();
#else
  return CompressionPrediction();
#endif
}

#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST

DMatrixHandle XGBoostPredictor::CreateDMatrix(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>* labels) {
  if (features.empty()) {
    return nullptr;
  }

  bst_ulong num_rows = static_cast<bst_ulong>(features.size());
  bst_ulong num_cols = static_cast<bst_ulong>(CompressionFeatures::NumFeatures());

  std::vector<float> data;
  data.reserve(num_rows * num_cols);
  for (const auto& f : features) {
    auto vec = f.ToVector();
    data.insert(data.end(), vec.begin(), vec.end());
  }

  DMatrixHandle dmatrix = nullptr;
  if (XGDMatrixCreateFromMat(data.data(), num_rows, num_cols, -1.0f,
                             &dmatrix) != 0 ||
      dmatrix == nullptr) {
    return nullptr;
  }

  if (labels != nullptr && !labels->empty()) {
    if (XGDMatrixSetFloatInfo(dmatrix, "label", labels->data(),
                              labels->size()) != 0) {
      XGDMatrixFree(dmatrix);
      return nullptr;
    }
  }

  return dmatrix;
}

void XGBoostPredictor::FreeDMatrix(DMatrixHandle dmatrix) {
  if (dmatrix != nullptr) {
    XGDMatrixFree(dmatrix);
  }
}

bool XGBoostPredictor::PredictWithBooster(BoosterHandle booster,
                                          DMatrixHandle dmatrix,
                                          std::vector<float>& out) {
  if (booster == nullptr || dmatrix == nullptr) {
    return false;
  }

  bst_ulong out_len = 0;
  const float* out_result = nullptr;
  if (XGBoosterPredict(booster, dmatrix, 0, 0, 0, &out_len, &out_result) !=
          0 ||
      out_result == nullptr || out_len == 0) {
    return false;
  }

  out.assign(out_result, out_result + out_len);
  return true;
}

std::vector<CompressionPrediction> XGBoostPredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());

  if (!is_ready_ || batch.empty()) {
    return results;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  DMatrixHandle dmatrix = CreateDMatrix(batch, nullptr);
  if (dmatrix == nullptr) {
    return results;
  }

  std::vector<float> ratio_preds;
  if (!PredictWithBooster(ratio_booster_, dmatrix, ratio_preds)) {
    FreeDMatrix(dmatrix);
    return results;
  }

  std::vector<float> psnr_preds(batch.size(), 0.0f);
  if (psnr_booster_ != nullptr) {
    PredictWithBooster(psnr_booster_, dmatrix, psnr_preds);
  }

  std::vector<float> time_preds(batch.size(), 0.0f);
  if (time_booster_ != nullptr) {
    PredictWithBooster(time_booster_, dmatrix, time_preds);
  }

  FreeDMatrix(dmatrix);

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  double per_sample_ms = total_ms / static_cast<double>(batch.size());

  for (size_t i = 0; i < batch.size(); ++i) {
    results.emplace_back(static_cast<double>(ratio_preds[i]),
                         static_cast<double>(psnr_preds[i]),
                         static_cast<double>(time_preds[i]), per_sample_ms);
  }

  return results;
}

#endif  // CTP_ENABLE_XGBOOST

}  // namespace ctp::compress::model
