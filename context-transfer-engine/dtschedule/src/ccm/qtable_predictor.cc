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

#include <clio_cte/dtschedule/ccm/qtable_predictor.h>

#include <clio_ctp/util/logging.h>
#include <fstream>
#include <algorithm>

namespace clio::cte::dtschedule::ccm {

/** Largest effective sample count for the online running mean. */
static constexpr double kMaxOnlineWeight = 16.0;

bool QtablePredictor::Load(const std::string &model_dir) {
  // Construct file paths
  std::string qtable_path = model_dir + "/qtable.json";
  std::string binning_path = model_dir + "/binning_params.json";

  // Load qtable.json
  std::ifstream qtable_file(qtable_path);
  if (!qtable_file.is_open()) {
    HLOG(kError, "QtablePredictor::Load failed to open {}", qtable_path);
    return false;
  }
  try {
    qtable_file >> qtable_;
  } catch (const std::exception &e) {
    HLOG(kError, "QtablePredictor::Load failed to parse qtable.json: {}", e.what());
    return false;
  }

  // Load binning_params.json
  std::ifstream binning_file(binning_path);
  if (!binning_file.is_open()) {
    HLOG(kError, "QtablePredictor::Load failed to open {}", binning_path);
    return false;
  }
  try {
    binning_file >> binning_params_;
  } catch (const std::exception &e) {
    HLOG(kError, "QtablePredictor::Load failed to parse binning_params.json: {}", e.what());
    return false;
  }

  // Parse encoding maps from qtable["encoding"]
  // Structure: encoding -> {library_id: {...}, config_id: {...}, datatype_id: {...}}
  if (qtable_.contains("encoding")) {
    auto &encoding = qtable_["encoding"];

    // Parse library_id map
    if (encoding.contains("library_id")) {
      auto &lib_map = encoding["library_id"];
      for (auto it = lib_map.begin(); it != lib_map.end(); ++it) {
        lib_to_id_[it.key()] = it.value().get<int>();
      }
    }

    // Parse datatype_id map (not dtype_id)
    if (encoding.contains("datatype_id")) {
      auto &dtype_map = encoding["datatype_id"];
      for (auto it = dtype_map.begin(); it != dtype_map.end(); ++it) {
        dtype_to_id_[it.key()] = it.value().get<int>();
      }
    }
  }

  // Build state_cache from states array for O(1) lookup
  size_t n_states_loaded = 0;
  if (qtable_.contains("states")) {
    auto &states = qtable_["states"];
    for (const auto &state_entry : states) {
      if (state_entry.contains("state")) {
        auto &state_vec = state_entry["state"];
        std::array<int, 7> state_key;
        for (size_t i = 0; i < 7; ++i) {
          state_key[i] = state_vec[i].get<int>();
        }

        // Read correct JSON field names: compression_time_ms, decompression_time_ms, compression_ratio
        double ctime = state_entry.contains("compression_time_ms")
                          ? state_entry["compression_time_ms"].get<double>()
                          : 1.0;
        double dtime = state_entry.contains("decompression_time_ms")
                          ? state_entry["decompression_time_ms"].get<double>()
                          : 1.0;
        double ratio = state_entry.contains("compression_ratio")
                          ? state_entry["compression_ratio"].get<double>()
                          : 1.0;

        double count = state_entry.contains("sample_count")
                           ? state_entry["sample_count"].get<double>()
                           : 1.0;
        if (size_only_) {
          // Fold every data-type/statistic bin into its size bin.
          state_key[2] = state_key[4] = state_key[5] = state_key[6] = 0;
          Entry &e = state_cache_[state_key];
          const double tot = e.count_ + count;
          e.mean_.ctime_ms_ = (e.mean_.ctime_ms_ * e.count_ + ctime * count) / tot;
          e.mean_.dtime_ms_ = (e.mean_.dtime_ms_ * e.count_ + dtime * count) / tot;
          e.mean_.ratio_ = (e.mean_.ratio_ * e.count_ + ratio * count) / tot;
          e.count_ = tot;
        } else {
          state_cache_[state_key] = Entry{Prediction{ctime, dtime, ratio}, count};
        }
        AccumulateMarginal({state_key[0], state_key[1], state_key[2]},
                           Prediction{ctime, dtime, ratio}, count);
        n_states_loaded++;
      }
    }
  }

  LoadEdges();
  HLOG(kInfo, "QtablePredictor::Load loaded {} states ({} codec marginals) from {}",
       n_states_loaded, marginal_.size(), qtable_path);
  return true;
}

Prediction QtablePredictor::Predict(const Features &features,
                                     const std::string &lib,
                                     ctp::CompressionPreset preset) {
  // Map library name to id
  auto lib_it = lib_to_id_.find(lib);
  if (lib_it == lib_to_id_.end()) {
    return NeutralPrior(features.size_);
  }
  int lib_id = lib_it->second;

  // Build state tuple
  auto state = BuildStateTuple(features, lib_id, preset);

  // O(1) lookup in state_cache
  auto it = state_cache_.find(state);
  if (it != state_cache_.end()) {
    return it->second.mean_;
  }
  // Unknown bin: fall back to this codec's mean over every bin it was
  // trained on (keeps the codec ranking meaningful), then to the prior.
  auto m = marginal_.find({state[0], state[1], state[2]});
  if (m != marginal_.end()) {
    return m->second.mean_;
  }
  return NeutralPrior(features.size_);
}

void QtablePredictor::Observe(const Features &features,
                              const std::string &lib,
                              ctp::CompressionPreset preset,
                              double obs_ctime_ms,
                              double obs_dtime_ms,
                              double obs_ratio) {
  auto lib_it = lib_to_id_.find(lib);
  if (lib_it == lib_to_id_.end()) {
    return;
  }
  auto state = BuildStateTuple(features, lib_it->second, preset);
  // Running mean on the table entry itself (created on first sight), so the
  // next Predict of this state sees the observation; the training count
  // weights how fast a state moves.
  // The training count is capped so a model trained on another machine
  // (or a stale one) converges to the observed values within a few tens of
  // observations instead of never moving.
  Entry &e = state_cache_[state];
  const double n = std::min(e.count_, kMaxOnlineWeight);
  e.mean_.ctime_ms_ = (e.mean_.ctime_ms_ * n + obs_ctime_ms) / (n + 1.0);
  // The write path observes no decompression (obs_dtime_ms <= 0 means
  // unknown): keep the learned decompression time instead of pulling it
  // toward zero.
  const double dtime = obs_dtime_ms > 0.0 ? obs_dtime_ms : e.mean_.dtime_ms_;
  e.mean_.dtime_ms_ = (e.mean_.dtime_ms_ * n + dtime) / (n + 1.0);
  e.mean_.ratio_ = (e.mean_.ratio_ * n + obs_ratio) / (n + 1.0);
  e.count_ = n + 1.0;
  AccumulateMarginal({state[0], state[1], state[2]},
                     Prediction{obs_ctime_ms, dtime, obs_ratio}, 1.0);
}

void QtablePredictor::AccumulateMarginal(const std::array<int, 3> &key,
                                         const Prediction &value,
                                         double count) {
  // The marginal is the fallback for states the table never saw: it keeps
  // its full training weight so a burst of one data kind cannot drag the
  // prediction for every other kind along (only exact states are capped).
  Entry &e = marginal_[key];
  const double n = e.count_;
  const double tot = n + count;
  e.mean_.ctime_ms_ = (e.mean_.ctime_ms_ * n + value.ctime_ms_ * count) / tot;
  e.mean_.dtime_ms_ = (e.mean_.dtime_ms_ * n + value.dtime_ms_ * count) / tot;
  e.mean_.ratio_ = (e.mean_.ratio_ * n + value.ratio_ * count) / tot;
  e.count_ = tot;
}

void QtablePredictor::LoadEdges() {
  // binning_params.json: bin_edges has 11 slots indexed by feature_slots
  // (size 3, entropy 4, mad 5, derivative 6), as train_qtable.py writes it.
  for (auto &v : edges_) v.clear();
  if (!binning_params_.contains("bin_edges")) return;
  const auto &edges = binning_params_["bin_edges"];
  const int slots[4] = {3, 4, 5, 6};
  for (int i = 0; i < 4; ++i) {
    if (static_cast<size_t>(slots[i]) < edges.size()) {
      for (const auto &e : edges[slots[i]]) {
        edges_[i].push_back(e.get<float>());
      }
    }
  }
}

std::array<int, 7> QtablePredictor::BuildStateTuple(
    const Features &features, int lib_id, ctp::CompressionPreset preset) const {
  // Map preset to config_id: kFast=1, kBalanced=2, kBest=3
  int config_id = 1;  // default to Fast
  switch (preset) {
    case ctp::CompressionPreset::FAST:
      config_id = 1;
      break;
    case ctp::CompressionPreset::BALANCED:
      config_id = 2;
      break;
    case ctp::CompressionPreset::BEST:
      config_id = 3;
      break;
    case ctp::CompressionPreset::DEFAULT:
      config_id = 2;  // default to balanced
      break;
  }

  // Map dtype to id: 0=char, 1=float, 2=int
  int dtype_id = features.dtype_;
  if (dtype_id < 0 || dtype_id > 2) {
    dtype_id = 0;  // default to char
  }

  // Bin continuous features using bin_edges
  const int size_bin = BinFeature(static_cast<double>(features.size_), edges_[0]);
  if (size_only_) {
    return {lib_id, config_id, 0, size_bin, 0, 0, 0};
  }
  const int entropy_bin = BinFeature(features.entropy_, edges_[1]);
  const int mad_bin = BinFeature(features.mad_, edges_[2]);
  const int d2_bin = BinFeature(features.d2_, edges_[3]);
  return {lib_id, config_id, dtype_id, size_bin, entropy_bin, mad_bin, d2_bin};
}

int QtablePredictor::BinFeature(double value,
                                 const std::vector<float> &edges) const {
  if (edges.empty()) {
    return 0;
  }

  // Find lower_bound: first edge >= value
  auto it = std::lower_bound(edges.begin(), edges.end(),
                              static_cast<float>(value));
  int bin_idx = std::distance(edges.begin(), it);

  // Clamp to valid range [0, edges.size()]
  bin_idx = std::max(0, std::min(bin_idx, static_cast<int>(edges.size())));

  return bin_idx;
}

}  // namespace clio::cte::dtschedule::ccm
