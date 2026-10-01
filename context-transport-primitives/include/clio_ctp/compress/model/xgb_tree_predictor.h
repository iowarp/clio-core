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
 * @file xgb_tree_predictor.h
 * @brief The XGBoost baseline, evaluated from exported trees at write time.
 *
 * The model is paper-benchmark/model-accuracy/train_xgb.py's: one regressor
 * per output (compression time, decompression time, ratio, PSNR), trained on
 * the same corpus rows that seed HCompress, over the NeuroPress NN's own
 * inputs. export_xgb_trees.py flattens it to xgb_trees.txt; this class walks
 * those trees, so clio-core carries no XGBoost dependency.
 *
 * It is a SELECTOR BASELINE, like HCompressCcpPredictor: the runtime ranks
 * NeuroPress's candidate set under NeuroPress's cost model with this model's
 * predictions in place of the network's, so the predictor is the only thing
 * that differs between the two arms.
 *
 * Load() evaluates the test vectors the exporter wrote (XGBoost's own
 * predict() on held-out rows) and refuses the model if any output differs by
 * more than 1e-4 -- a wrong feature order or encoding cannot go unnoticed.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_XGB_TREE_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_XGB_TREE_PREDICTOR_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/predictor.h"

namespace ctp::compress::model {

/** Number of model inputs: 8 one-hot algorithms, quantize, shuffle, log10 eb,
 *  log2 bytes, entropy, MAD, second derivative. */
constexpr size_t kXgbNumFeatures = 15;

/** Outputs, in xgb_trees.txt order. */
enum XgbOutput : int {
  kXgbCompTime = 0,
  kXgbDecompTime = 1,
  kXgbRatio = 2,
  kXgbPsnr = 3,
  kXgbNumOutputs = 4,
};

/** One output's ensemble, its trees' nodes stored back to back. */
struct XgbEnsemble {
  double y_mean = 0.0;      /**< undo the training target's standardization */
  double y_std = 1.0;
  double base_score = 0.0;  /**< XGBoost's global bias, added once */
  std::vector<int32_t> tree_offset;  /**< first node of each tree */
  std::vector<int32_t> left;         /**< -1 marks a leaf */
  std::vector<int32_t> right;
  std::vector<int32_t> missing;      /**< taken when the input is NaN */
  std::vector<int32_t> feature;
  std::vector<float> threshold;      /**< go left when x < threshold (float32) */
  std::vector<float> leaf;
};

/**
 * @brief XGBoost regressors evaluated from exported, flattened trees.
 */
class XgbTreePredictor : public CompressionPredictor {
 public:
  /**
   * @brief Load `model_dir`/xgb_trees.txt and verify it against its own
   * test vectors.
   * @param model_dir directory holding xgb_trees.txt
   * @return true when the file parsed and every test vector matched
   */
  bool Load(const std::string &model_dir) override;

  /** Read-only model: trained offline by train_xgb.py. Always false. */
  bool Save(const std::string &model_dir) override;

  /** True once Load() succeeded. */
  bool IsReady() const override { return ready_; }

  /** ModelType::kXGBoost. */
  ModelType Type() const override { return ModelType::kXGBoost; }

  /**
   * @brief Predict one candidate's metrics.
   * @param features the chunk's statistics plus the candidate's library,
   *        shuffle, quantize and bound (MakeCompressionFeatures)
   * @return compression/decompression time (ms), ratio and PSNR (dB)
   */
  CompressionPrediction Predict(const CompressionFeatures &features) override;

  /** Why the last Load() failed; empty after a success. */
  const std::string &LastError() const { return error_; }

  /** Trees per output, for the startup log. */
  size_t NumTrees(int output) const {
    return ensembles_[output].tree_offset.size();
  }

  /** Test vectors Load() checked. */
  size_t NumTestVectors() const { return tests_checked_; }

 private:
  /** Standardize raw inputs and walk every ensemble.
   *  @param raw  the 15 unstandardized inputs
   *  @param out  each output in the TRAINING target's standardized space */
  void EvalRaw(const std::array<double, kXgbNumFeatures> &raw,
               std::array<double, kXgbNumOutputs> *out) const;

  /** Sum one ensemble's leaves for a standardized float32 input. */
  static double EvalEnsemble(const XgbEnsemble &e, const float *x);

  /** Encode a candidate the way train_xgb.py's encode_x does. */
  std::array<double, kXgbNumFeatures> Encode(
      const CompressionFeatures &f) const;

  /** Parse xgb_trees.txt; fills tests with (raw inputs, expected outputs). */
  bool Parse(const std::string &path,
             std::vector<std::array<double, kXgbNumFeatures + kXgbNumOutputs>>
                 *tests);

  std::array<double, kXgbNumFeatures> x_means_{};
  std::array<double, kXgbNumFeatures> x_stds_{};
  double lossless_eb_ = 1e-7;
  std::array<XgbEnsemble, kXgbNumOutputs> ensembles_;
  bool ready_ = false;
  size_t tests_checked_ = 0;
  std::string error_;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_XGB_TREE_PREDICTOR_H_
