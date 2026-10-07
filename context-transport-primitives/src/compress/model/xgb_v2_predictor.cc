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

#include "clio_ctp/compress/model/xgb_v2_predictor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <cctype>

#include "clio_ctp/compress/gpu_setting_codec.h"

namespace ctp::compress::model {

namespace {

/** Largest |exported predict() - ours| Load() accepts on a test vector. */
constexpr double kTestTolerance = 1e-4;

/** Read `keyword` from `in`; false when the next token is anything else. */
bool Expect(std::istream &in, const char *keyword) {
  std::string tok;
  return static_cast<bool>(in >> tok) && tok == keyword;
}

}  // namespace

bool XgbV2Predictor::Parse(
    const std::string &path,
    std::vector<std::array<double, kXgbV2NumChunkFeatures + kXgbV2NumSettings * kXgbV2NumOutputs>> *tests) {
  std::ifstream in(path);
  if (!in) { error_ = "cannot open " + path; return false; }
  int version = 0;
  if (!Expect(in, "clio-xgb-v2-trees") || !(in >> version) || version != 1) {
    error_ = path + ": not a version-1 clio-xgb-v2-trees file";
    return false;
  }
  size_t nf = 0;
  if (!Expect(in, "chunk_features") || !(in >> nf) || nf != kXgbV2NumChunkFeatures) {
    error_ = "expected 4 chunk features";
    return false;
  }
  size_t ns = 0;
  if (!Expect(in, "settings") || !(in >> ns) || ns != kXgbV2NumSettings) {
    error_ = "expected 45 settings";
    return false;
  }
  // One spec per line; a spec has spaces ("ans shuffle=bit"), so read lines.
  std::vector<std::string> specs(ns);
  std::string rest;
  std::getline(in, rest);  // the end of the "settings <S>" line
  for (size_t i = 0; i < ns; ++i) {
    if (!std::getline(in, specs[i])) { error_ = "truncated settings"; return false; }
    while (!specs[i].empty() && std::isspace(static_cast<unsigned char>(specs[i].back()))) {
      specs[i].pop_back();
    }
    if (specs[i].empty()) { error_ = "empty setting spec"; return false; }
  }
  if (!MapSettingSpecs(specs)) return false;

  size_t no = 0;
  if (!Expect(in, "outputs") || !(in >> no) || no != kXgbV2NumOutputs) {
    error_ = "expected 3 outputs";
    return false;
  }
  for (size_t k = 0; k < no; ++k) {
    std::string name;
    size_t n_trees = 0;
    XgbV2Ensemble &e = ensembles_[k];
    if (!Expect(in, "output") || !(in >> name) || !(in >> e.base_score >> n_trees)) {
      error_ = "bad header for output " + std::to_string(k);
      return false;
    }
    for (size_t t = 0; t < n_trees; ++t) {
      size_t n_nodes = 0;
      if (!Expect(in, "tree") || !(in >> n_nodes) || n_nodes == 0) {
        error_ = "bad tree " + std::to_string(t) + " for output " + std::to_string(k);
        return false;
      }
      const int32_t base = static_cast<int32_t>(e.left.size());
      e.tree_offset.push_back(base);
      for (size_t n = 0; n < n_nodes; ++n) {
        int32_t l, r, m, f;
        double thr, leaf;
        if (!(in >> l >> r >> m >> f >> thr >> leaf) ||
            (l >= 0 && (f < 0 || f >= static_cast<int32_t>(kXgbV2NumFeatures) ||
                        l >= static_cast<int32_t>(n_nodes) ||
                        r >= static_cast<int32_t>(n_nodes) ||
                        m >= static_cast<int32_t>(n_nodes)))) {
          error_ = "bad node in tree " + std::to_string(t);
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

bool XgbV2Predictor::MapSettingSpecs(const std::vector<std::string> &specs) {
  if (specs.size() != kXgbV2NumSettings) {
    error_ = "expected " + std::to_string(kXgbV2NumSettings) + " settings";
    return false;
  }
  for (size_t i = 0; i < specs.size(); ++i) {
    int idx = GpuSettingIndex(specs[i]);
    if (idx < 0) {
      error_ = "unknown setting spec: " + specs[i];
      return false;
    }
    setting_map_[i] = static_cast<int32_t>(idx);
  }
  return true;
}

void XgbV2Predictor::EvalEnsembleWithMask(
    const XgbV2Ensemble &e, const float x[kXgbV2NumChunkFeatures],
    double *out) const {
  // Initialize output for all settings.
  for (size_t s = 0; s < kXgbV2NumSettings; ++s) {
    out[s] = e.base_score;
  }

  // Walk each tree with a 64-bit mask of settings that reach nodes.
  // Using 64-bit mask is safe since kXgbV2NumSettings = 45 < 64.
  constexpr uint64_t all_settings = (1ULL << kXgbV2NumSettings) - 1;

  for (int32_t root : e.tree_offset) {
    // Stack: (node_index, mask of settings at this node)
    std::array<std::pair<int32_t, uint64_t>, 128> stack;
    int stack_top = 0;
    stack[stack_top++] = {root, all_settings};

    while (stack_top > 0) {
      const auto [n, mask] = stack[--stack_top];
      if (mask == 0) continue;  // No settings at this node.

      // Leaf node: accumulate leaf value for all settings in the mask.
      if (e.left[n] < 0) {
        for (size_t s = 0; s < kXgbV2NumSettings; ++s) {
          if ((mask >> s) & 1) {
            out[s] += e.leaf[n];
          }
        }
        continue;
      }

      const int32_t f = e.feature[n];

      // Chunk feature node: all settings follow the same child.
      if (f < kXgbV2NumChunkFeatures) {
        const float v = x[f];
        const int32_t nxt = std::isnan(v) ? e.missing[n]
                                          : (v < e.threshold[n] ? e.left[n] : e.right[n]);
        stack[stack_top++] = {nxt, mask};
        continue;
      }

      // One-hot node: feature f corresponds to setting (f - kXgbV2NumChunkFeatures).
      const size_t model_setting = f - kXgbV2NumChunkFeatures;
      if (model_setting >= kXgbV2NumSettings) continue;  // Malformed.

      // Determine where 1 and 0 go.
      const int32_t one_child = static_cast<float>(1.0f) < e.threshold[n] ? e.left[n] : e.right[n];
      const int32_t zero_child = static_cast<float>(0.0f) < e.threshold[n] ? e.left[n] : e.right[n];

      if (one_child == zero_child) {
        // Both 1 and 0 go the same way; no split.
        stack[stack_top++] = {one_child, mask};
      } else {
        // Split: bit model_setting goes to one_child, the rest to zero_child.
        const uint64_t bit_mask = 1ULL << model_setting;
        if (mask & bit_mask) {
          stack[stack_top++] = {one_child, mask & bit_mask};
        }
        if (mask & ~bit_mask) {
          stack[stack_top++] = {zero_child, mask & ~bit_mask};
        }
      }
    }
  }
}

bool XgbV2Predictor::Load(const std::string &model_dir) {
  ready_ = false;
  tests_checked_ = 0;
  error_.clear();
  for (auto &e : ensembles_) e = XgbV2Ensemble{};
  std::fill(setting_map_.begin(), setting_map_.end(), -1);

  std::vector<std::array<double, kXgbV2NumChunkFeatures + kXgbV2NumSettings * kXgbV2NumOutputs>> tests;
  if (!Parse(model_dir + "/xgb_v2_trees.txt", &tests)) return false;
  if (tests.empty()) {
    error_ = "no test vectors: the model cannot be verified";
    return false;
  }

  // Evaluate test vectors.
  for (size_t t = 0; t < tests.size(); ++t) {
    // Convert chunk features to float32.
    float x[kXgbV2NumChunkFeatures];
    for (size_t i = 0; i < kXgbV2NumChunkFeatures; ++i) {
      x[i] = static_cast<float>(tests[t][i]);
    }

    // Predict using all ensembles.
    double got[kXgbV2NumSettings * kXgbV2NumOutputs];
    for (int k = 0; k < kXgbV2NumOutputs; ++k) {
      double out[kXgbV2NumSettings];
      EvalEnsembleWithMask(ensembles_[k], x, out);
      for (size_t s = 0; s < kXgbV2NumSettings; ++s) {
        got[s * kXgbV2NumOutputs + k] = out[s];   // setting-major, as the file
      }
    }

    // Check against expected values (setting-major order: s0 comp, s0 decomp, s0 ratio, s1 comp, ...).
    const double *want_ptr = &tests[t][kXgbV2NumChunkFeatures];
    for (size_t i = 0; i < kXgbV2NumSettings * kXgbV2NumOutputs; ++i) {
      const double want = want_ptr[i];
      if (!(std::fabs(got[i] - want) <= kTestTolerance)) {
        std::ostringstream os;
        os << "test vector " << t << ", output " << (i % kXgbV2NumOutputs)
           << ", setting " << (i / kXgbV2NumOutputs) << ": got " << got[i]
           << ", expected " << want << " (diff=" << std::fabs(got[i] - want) << ")";
        error_ = os.str();
        return false;
      }
    }
  }

  tests_checked_ = tests.size();
  ready_ = true;
  return true;
}

void XgbV2Predictor::PredictAll(const float x[kXgbV2NumChunkFeatures],
                                 double comp_ms_log[kXgbV2NumSettings],
                                 double decomp_ms_log[kXgbV2NumSettings],
                                 double ratio_log[kXgbV2NumSettings]) const {
  // Initialize with NaN (setting not in model or error).
  const double nan_val = std::numeric_limits<double>::quiet_NaN();
  std::fill(comp_ms_log, comp_ms_log + kXgbV2NumSettings, nan_val);
  std::fill(decomp_ms_log, decomp_ms_log + kXgbV2NumSettings, nan_val);
  std::fill(ratio_log, ratio_log + kXgbV2NumSettings, nan_val);

  if (!ready_) return;

  // Evaluate each ensemble.
  double comp_out[kXgbV2NumSettings];
  double decomp_out[kXgbV2NumSettings];
  double ratio_out[kXgbV2NumSettings];

  EvalEnsembleWithMask(ensembles_[kXgbV2CompTime], x, comp_out);
  EvalEnsembleWithMask(ensembles_[kXgbV2DecompTime], x, decomp_out);
  EvalEnsembleWithMask(ensembles_[kXgbV2Ratio], x, ratio_out);

  // Map the model's setting m (one-hot order) to Clio's setting index.
  for (size_t m = 0; m < kXgbV2NumSettings; ++m) {
    const int32_t c = setting_map_[m];
    if (c < 0 || c >= static_cast<int32_t>(kXgbV2NumSettings)) continue;
    comp_ms_log[c] = comp_out[m];
    decomp_ms_log[c] = decomp_out[m];
    ratio_log[c] = ratio_out[m];
  }
}

}  // namespace ctp::compress::model
