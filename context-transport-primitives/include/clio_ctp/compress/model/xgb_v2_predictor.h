/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file xgb_v2_predictor.h
 * @brief XGBoost v2 selector for NeuroPress v2: ranks 45 GPU settings with
 * XGBoost predictions instead of the NeuroPress v2 network.
 *
 * The model is paper-benchmark/model-accuracy/train_xgb_v2.py's: one regressor
 * per output (compression time, decompression time, ratio), trained on the
 * same corpus used by NeuroPress v2, over the 4 NeuroPress v2 chunk features
 * (log2 bytes, entropy, MAD, log10 2nd derivative) plus one-hot of the setting.
 * export_xgb_v2_trees.py flattens it to xgb_v2_trees.txt; this class walks
 * those trees, so clio-core carries no XGBoost dependency.
 *
 * It is a SELECTOR BASELINE: the runtime ranks the 45 GPU settings under the
 * NeuroPress v2 cost model with this model's predictions in place of the
 * network's, so the predictor is the only thing that differs from NeuroPress v2.
 *
 * Load() evaluates the test vectors the exporter wrote (XGBoost's own
 * predict() on held-out rows) and refuses the model if any output differs by
 * more than 1e-4 -- a wrong feature order or encoding cannot go unnoticed.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_XGB_V2_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_XGB_V2_PREDICTOR_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ctp::compress::model {

/** Number of chunk features (before one-hot): log2 bytes, entropy, MAD,
 *  log10 second derivative. */
constexpr size_t kXgbV2NumChunkFeatures = 4;

/** Number of GPU settings in the model (one-hot length). */
constexpr size_t kXgbV2NumSettings = 45;

/** Total model features: 4 chunk + 45 one-hot. */
constexpr size_t kXgbV2NumFeatures =
    kXgbV2NumChunkFeatures + kXgbV2NumSettings;

/** Outputs, in xgb_v2_trees.txt order. */
enum XgbV2Output : int {
  kXgbV2CompTime = 0,    ///< log(compress ms)
  kXgbV2DecompTime = 1,  ///< log(decompress ms)
  kXgbV2Ratio = 2,       ///< log(compression ratio)
  kXgbV2NumOutputs = 3,
};

/** One output's ensemble, its trees' nodes stored back to back. */
struct XgbV2Ensemble {
  double base_score = 0.0;   /**< XGBoost's global bias, added once */
  std::vector<int32_t> tree_offset;  /**< first node of each tree */
  std::vector<int32_t> left;         /**< -1 marks a leaf */
  std::vector<int32_t> right;
  std::vector<int32_t> missing;      /**< taken when the input is NaN */
  std::vector<int32_t> feature;
  std::vector<float> threshold;      /**< go left when x < threshold (float32) */
  std::vector<float> leaf;
};

/**
 * @brief XGBoost v2 regressors evaluated from exported, flattened trees.
 *
 * Efficiently walks the trees with a 64-bit mask of settings that reach each
 * node, honoring the constraint that only one setting's one-hot can differ at
 * one-hot nodes. Outputs are natural logarithms (exp() them).
 */
class XgbV2Predictor {
 public:
  /**
   * @brief Load `model_dir`/xgb_v2_trees.txt, map setting specs to indices,
   * and verify against its own test vectors.
   * @param model_dir directory holding xgb_v2_trees.txt
   * @return true when the file parsed, settings matched, and every test vector
   *         matched
   */
  bool Load(const std::string &model_dir);

  /** True once Load() succeeded. */
  bool IsReady() const { return ready_; }

  /**
   * @brief Predict all 45 settings' metrics from the 4 chunk features.
   *
   * Outputs are natural logarithms. Uses efficient tree walking with a
   * 64-bit mask of settings that reach each node. Thread-safe (no heap
   * allocation, const).
   *
   * @param x the 4 chunk features (log2 bytes, entropy, MAD, log10 2nd deriv)
   * @param comp_ms_log output: natural log compress time for each setting (NaN
   *                    for settings not in the model)
   * @param decomp_ms_log output: natural log decompress time for each setting
   * @param ratio_log output: natural log compression ratio for each setting
   */
  void PredictAll(const float x[kXgbV2NumChunkFeatures],
                  double comp_ms_log[kXgbV2NumSettings],
                  double decomp_ms_log[kXgbV2NumSettings],
                  double ratio_log[kXgbV2NumSettings]) const;

  /** Why the last Load() failed; empty after a success. */
  const std::string &LastError() const { return error_; }

  /** Trees per output, for the startup log. */
  size_t NumTrees(int output) const {
    return ensembles_[output].tree_offset.size();
  }

  /** Test vectors Load() checked. */
  size_t NumTestVectors() const { return tests_checked_; }

 private:
  /** Parse xgb_v2_trees.txt; fills tests with (4 chunk features, 45*3 predictions). */
  bool Parse(const std::string &path,
             std::vector<std::array<double, kXgbV2NumChunkFeatures + kXgbV2NumSettings * kXgbV2NumOutputs>> *tests);

  /** Map setting specs to Clio's GpuSettingIndex; false if any spec is unknown. */
  bool MapSettingSpecs(const std::vector<std::string> &specs);

  /** Walk a single ensemble with a mask of settings, accumulating predictions.
   *  @param e the ensemble
   *  @param x the 4 chunk features (as float32)
   *  @param out per-setting predictions (output index * 45 + setting) */
  void EvalEnsembleWithMask(const XgbV2Ensemble &e, const float x[kXgbV2NumChunkFeatures],
                            double *out) const;

  std::array<XgbV2Ensemble, kXgbV2NumOutputs> ensembles_;
  /** Maps model setting index (0-44) to Clio setting index (0-255). */
  std::array<int32_t, kXgbV2NumSettings> setting_map_;
  bool ready_ = false;
  size_t tests_checked_ = 0;
  std::string error_;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_XGB_V2_PREDICTOR_H_
