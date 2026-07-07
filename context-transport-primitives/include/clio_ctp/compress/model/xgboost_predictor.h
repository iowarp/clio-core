/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file xgboost_predictor.h
 * @brief XGBoost-based compression-metric predictor (optional external dep).
 *
 * Implements CompressionPredictor using XGBoost gradient boosting for
 * compression ratio, PSNR, and inference-time prediction. When
 * CTP_ENABLE_XGBOOST is undefined or 0, the class compiles but IsReady()
 * returns false and Predict() yields zero predictions.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_XGBOOST_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_XGBOOST_PREDICTOR_H_

#include "clio_ctp/compress/model/predictor.h"
#include <mutex>
#include <string>
#include <vector>

#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
#include <xgboost/c_api.h>
#endif

namespace ctp::compress::model {

/**
 * @brief XGBoost gradient-boosting predictor for compression metrics.
 *
 * Maintains three internal boosters (one per output: ratio, PSNR, time) when
 * enabled. When CTP_ENABLE_XGBOOST is off, it is a no-op that compiles and
 * links without xgboost library.
 */
class XGBoostPredictor : public CompressionPredictor {
 public:
  /** @brief Default constructor. */
  XGBoostPredictor();

  /** @brief Destructor frees xgboost resources (if enabled). */
  ~XGBoostPredictor() override;

  // Disable copy (xgboost handles are not copyable).
  XGBoostPredictor(const XGBoostPredictor&) = delete;
  XGBoostPredictor& operator=(const XGBoostPredictor&) = delete;

  // Enable move.
  XGBoostPredictor(XGBoostPredictor&& other) noexcept;
  XGBoostPredictor& operator=(XGBoostPredictor&& other) noexcept;

  /**
   * @brief Load three JSON booster models from model_dir.
   *
   * Expects:
   * - compression_ratio_model.json
   * - psnr_model.json (optional)
   * - compression_time_model.json (optional)
   *
   * @param model_dir Directory containing model files.
   * @return false when CTP_ENABLE_XGBOOST is off.
   */
  bool Load(const std::string& model_dir) override;

  /**
   * @brief Persist booster models to model_dir as JSON.
   * @param model_dir Directory to write model files.
   * @return false when CTP_ENABLE_XGBOOST is off or not ready.
   */
  bool Save(const std::string& model_dir) override;

  /**
   * @brief True if boosters are loaded and ready.
   * @return false when CTP_ENABLE_XGBOOST is off.
   */
  bool IsReady() const override;

  /**
   * @brief Return ModelType::kXGBoost.
   */
  ModelType Type() const override {
    return ModelType::kXGBoost;
  }

  /**
   * @brief Predict compression metrics for one feature vector.
   * @param features Input features.
   * @return Zero prediction when CTP_ENABLE_XGBOOST is off.
   */
  CompressionPrediction Predict(
      const CompressionFeatures& features) override;

 private:
#if defined(CTP_ENABLE_XGBOOST) && CTP_ENABLE_XGBOOST
  using BoosterHandle = BoosterHandle;
  using DMatrixHandle = DMatrixHandle;

  // Booster handles (three separate models for multi-output).
  BoosterHandle ratio_booster_ = nullptr;
  BoosterHandle psnr_booster_ = nullptr;
  BoosterHandle time_booster_ = nullptr;
  bool is_ready_ = false;
  mutable std::mutex mutex_;

  // Helper to create a DMatrix from features + optional labels.
  DMatrixHandle CreateDMatrix(
      const std::vector<CompressionFeatures>& features,
      const std::vector<float>* labels = nullptr);

  // Helper to free a DMatrix.
  void FreeDMatrix(DMatrixHandle dmatrix);

  // Helper to predict with a booster.
  bool PredictWithBooster(BoosterHandle booster, DMatrixHandle dmatrix,
                          std::vector<float>& out_predictions);

  // Helper to run batch inference.
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch);
#endif
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_XGBOOST_PREDICTOR_H_
