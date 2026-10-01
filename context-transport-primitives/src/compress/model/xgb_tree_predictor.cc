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

#include "clio_ctp/compress/model/xgb_tree_predictor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"

namespace ctp::compress::model {

namespace {

/** The exporter's feature order; a file in any other order is refused. */
const char *const kFeatureNames[kXgbNumFeatures] = {
    "alg_lz4",      "alg_snappy",      "alg_deflate",     "alg_gdeflate",
    "alg_zstd",     "alg_ans",         "alg_cascaded",    "alg_bitcomp",
    "quant_enc",    "shuffle_enc",     "error_bound_enc", "data_size_enc",
    "entropy",      "mad",             "second_derivative"};

/** The exporter's output order; a file in any other order is refused. */
const char *const kOutputNames[kXgbNumOutputs] = {
    "comp_time_log", "decomp_time_log", "ratio_log", "psnr_clamped"};

/** Largest |exported predict() - ours| Load() accepts on a test vector. */
constexpr double kTestTolerance = 1e-4;

/** Read `keyword` from `in`; false when the next token is anything else. */
bool Expect(std::istream &in, const char *keyword) {
  std::string tok;
  return static_cast<bool>(in >> tok) && tok == keyword;
}

}  // namespace

bool XgbTreePredictor::Parse(
    const std::string &path,
    std::vector<std::array<double, kXgbNumFeatures + kXgbNumOutputs>> *tests) {
  std::ifstream in(path);
  if (!in) { error_ = "cannot open " + path; return false; }
  int version = 0;
  if (!Expect(in, "clio-xgb-trees") || !(in >> version) || version != 1) {
    error_ = path + ": not a version-1 clio-xgb-trees file";
    return false;
  }
  size_t nf = 0;
  if (!Expect(in, "features") || !(in >> nf) || nf != kXgbNumFeatures) {
    error_ = "expected 15 features";
    return false;
  }
  for (size_t i = 0; i < nf; ++i) {
    std::string name;
    if (!(in >> name) || name != kFeatureNames[i]) {
      error_ = "feature " + std::to_string(i) + " is '" + name +
               "', expected '" + kFeatureNames[i] + "'";
      return false;
    }
  }
  if (!Expect(in, "x_means")) { error_ = "missing x_means"; return false; }
  for (auto &v : x_means_) in >> v;
  if (!Expect(in, "x_stds")) { error_ = "missing x_stds"; return false; }
  for (auto &v : x_stds_) in >> v;
  if (!Expect(in, "lossless_eb") || !(in >> lossless_eb_) ||
      !(lossless_eb_ > 0.0)) {
    error_ = "missing or non-positive lossless_eb";
    return false;
  }
  size_t no = 0;
  if (!Expect(in, "outputs") || !(in >> no) || no != kXgbNumOutputs) {
    error_ = "expected 4 outputs";
    return false;
  }
  for (size_t k = 0; k < no; ++k) {
    std::string name;
    size_t n_trees = 0;
    XgbEnsemble &e = ensembles_[k];
    if (!Expect(in, "output") || !(in >> name) || name != kOutputNames[k] ||
        !(in >> e.y_mean >> e.y_std >> e.base_score >> n_trees)) {
      error_ = "bad header for output " + std::to_string(k);
      return false;
    }
    for (size_t t = 0; t < n_trees; ++t) {
      size_t n_nodes = 0;
      if (!Expect(in, "tree") || !(in >> n_nodes) || n_nodes == 0) {
        error_ = name + ": bad tree " + std::to_string(t);
        return false;
      }
      const int32_t base = static_cast<int32_t>(e.left.size());
      e.tree_offset.push_back(base);
      for (size_t n = 0; n < n_nodes; ++n) {
        int32_t l, r, m, f;
        double thr, leaf;
        if (!(in >> l >> r >> m >> f >> thr >> leaf) ||
            (l >= 0 && (f < 0 || f >= static_cast<int32_t>(kXgbNumFeatures) ||
                        l >= static_cast<int32_t>(n_nodes) ||
                        r >= static_cast<int32_t>(n_nodes) ||
                        m >= static_cast<int32_t>(n_nodes)))) {
          error_ = name + ": bad node in tree " + std::to_string(t);
          return false;
        }
        // Child indices are tree-local in the file; store them absolute.
        e.left.push_back(l < 0 ? -1 : base + l);
        e.right.push_back(r < 0 ? -1 : base + r);
        e.missing.push_back(m < 0 ? -1 : base + m);
        e.feature.push_back(f);
        e.threshold.push_back(static_cast<float>(thr));
        e.leaf.push_back(static_cast<float>(leaf));
      }
    }
  }
  size_t nt = 0;
  if (!Expect(in, "tests") || !(in >> nt)) {
    error_ = "missing test vectors";
    return false;
  }
  tests->resize(nt);
  for (auto &row : *tests) {
    for (auto &v : row) {
      if (!(in >> v)) { error_ = "truncated test vectors"; return false; }
    }
  }
  return true;
}

double XgbTreePredictor::EvalEnsemble(const XgbEnsemble &e, const float *x) {
  // XGBoost accumulates leaves in float32; summing in double differs by a few
  // ulps, far inside the load-time tolerance.
  double sum = e.base_score;
  for (int32_t root : e.tree_offset) {
    int32_t i = root;
    while (e.left[i] >= 0) {
      const float v = x[e.feature[i]];
      i = std::isnan(v) ? e.missing[i]
                        : (v < e.threshold[i] ? e.left[i] : e.right[i]);
    }
    sum += e.leaf[i];
  }
  return sum;
}

void XgbTreePredictor::EvalRaw(const std::array<double, kXgbNumFeatures> &raw,
                               std::array<double, kXgbNumOutputs> *out) const {
  // Standardized in double, compared in float32: what the Python side did
  // before handing XGBoost its float32 matrix.
  float x[kXgbNumFeatures];
  for (size_t i = 0; i < kXgbNumFeatures; ++i) {
    x[i] = static_cast<float>((raw[i] - x_means_[i]) / x_stds_[i]);
  }
  for (int k = 0; k < kXgbNumOutputs; ++k) {
    (*out)[k] = EvalEnsemble(ensembles_[k], x);
  }
}

std::array<double, kXgbNumFeatures> XgbTreePredictor::Encode(
    const CompressionFeatures &f) const {
  std::array<double, kXgbNumFeatures> raw{};
  // library_config_id = base_id * 10 + preset; the algorithm index is the
  // NN's own, so both models read the same 8-way encoding.
  const int base_id = static_cast<int>(f.library_config_id) / 10;
  const int algo = NeuroPressAlgoIdForBaseId(base_id);
  if (algo >= 0 && algo < 8) raw[algo] = 1.0;
  const bool quant = f.quantize > 0.0;
  raw[8] = quant ? 1.0 : 0.0;
  raw[9] = f.byte_shuffle > 0.0 ? 1.0 : 0.0;
  // Lossless rows trained at a fixed 1e-7: the corpus bound means nothing
  // for a candidate that does not quantize.
  const double eb = (quant && f.error_bound > 0.0) ? f.error_bound
                                                   : lossless_eb_;
  raw[10] = std::log10(eb);
  raw[11] = std::log2(std::max(1.0, f.chunk_size_bytes));
  raw[12] = f.shannon_entropy;
  raw[13] = f.mad;
  raw[14] = f.second_derivative_mean;
  return raw;
}

bool XgbTreePredictor::Load(const std::string &model_dir) {
  ready_ = false;
  tests_checked_ = 0;
  error_.clear();
  for (auto &e : ensembles_) e = XgbEnsemble{};
  std::vector<std::array<double, kXgbNumFeatures + kXgbNumOutputs>> tests;
  if (!Parse(model_dir + "/xgb_trees.txt", &tests)) return false;
  if (tests.empty()) {
    error_ = "no test vectors: the model cannot be verified";
    return false;
  }
  for (size_t t = 0; t < tests.size(); ++t) {
    std::array<double, kXgbNumFeatures> raw;
    for (size_t i = 0; i < kXgbNumFeatures; ++i) raw[i] = tests[t][i];
    std::array<double, kXgbNumOutputs> got;
    EvalRaw(raw, &got);
    for (int k = 0; k < kXgbNumOutputs; ++k) {
      const double want = tests[t][kXgbNumFeatures + k];
      if (!(std::fabs(got[k] - want) <= kTestTolerance)) {
        std::ostringstream os;
        os << "test vector " << t << ", " << kOutputNames[k] << ": got "
           << got[k] << ", XGBoost predict() gave " << want;
        error_ = os.str();
        return false;
      }
    }
  }
  tests_checked_ = tests.size();
  ready_ = true;
  return true;
}

bool XgbTreePredictor::Save(const std::string &model_dir) {
  (void)model_dir;
  error_ = "XgbTreePredictor is read-only; train with train_xgb.py";
  return false;
}

CompressionPrediction XgbTreePredictor::Predict(
    const CompressionFeatures &features) {
  const auto t0 = std::chrono::steady_clock::now();
  CompressionPrediction p;
  if (!ready_) return p;
  std::array<double, kXgbNumOutputs> y;
  EvalRaw(Encode(features), &y);
  auto unstd = [&](int k) {
    return y[k] * ensembles_[k].y_std + ensembles_[k].y_mean;
  };
  // Targets were log1p of ms and of the ratio; PSNR was used as-is.
  p.compression_time_ms = std::expm1(unstd(kXgbCompTime));
  p.decompression_time_ms = std::expm1(unstd(kXgbDecompTime));
  p.compression_ratio = std::expm1(unstd(kXgbRatio));
  p.psnr_db = unstd(kXgbPsnr);
  p.inference_time_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
  return p;
}

}  // namespace ctp::compress::model
