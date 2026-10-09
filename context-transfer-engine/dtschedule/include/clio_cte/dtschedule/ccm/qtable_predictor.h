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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_QTABLE_PREDICTOR_H_
#define CLIO_CTE_DTSCHEDULE_CCM_QTABLE_PREDICTOR_H_

#include <clio_cte/dtschedule/ccm/predictor.h>
#include <nlohmann/json.hpp>
#include <map>
#include <unordered_map>
#include <vector>
#include <array>
#include <string>

namespace clio::cte::dtschedule::ccm {

/**
 * Q-table predictor: loads trained model and uses running-mean updates.
 *
 * Loads qtable_v1/qtable.json and binning_params.json, builds state tuples
 * from (library_id, config_id, dtype_id, size_bin, entropy_bin, mad_bin, d2_bin),
 * looks up predictions, and online-updates the values with observed data
 * (incremental mean).
 *
 * Unknown states fall back to global_average from the model.
 */
class QtablePredictor : public CcmPredictor {
 public:
  QtablePredictor() = default;
  /**
   * @param size_only Key states by (codec, preset, size bin) only, folding
   *        the data-type and data-statistic bins together: the HCompress
   *        baseline, which predicts the best codec from the data size alone.
   */
  explicit QtablePredictor(bool size_only) : size_only_(size_only) {}
  /** Frozen table: predictions come from the trained table only (no online learning). */
  void SetFrozen(bool frozen) { frozen_ = frozen; }
  ~QtablePredictor() override = default;

  /**
   * Load the Q-table model from JSON files.
   *
   * Reads qtable.json (state values) and binning_params.json (bin edges)
   * from model_dir. Computes library_id, config_id, and dtype_id mappings.
   *
   * @param model_dir Directory containing qtable.json and binning_params.json
   * @return True if both files loaded and parsed successfully
   */
  bool Load(const std::string &model_dir);

  /**
   * Predict compression characteristics using the loaded Q-table.
   *
   * Builds state tuple from features and (lib, preset), looks it up,
   * and returns the table value. Unknown states return global_average.
   */
  Prediction Predict(const Features &features,
                     const std::string &lib,
                     ctp::CompressionPreset preset) override;

  /**
   * Update online running mean for a state.
   *
   * Increments sample count and updates mean ctime, dtime, ratio for the
   * state. Creates the state if absent. Supports resampling by returning
   * true if |pred - obs| / obs exceeds resample_error.
   */
  void Observe(const Features &features,
               const std::string &lib,
               ctp::CompressionPreset preset,
               double obs_ctime_ms,
               double obs_dtime_ms,
               double obs_ratio) override;

 private:
  /**
   * Build state tuple: [lib_id, config_id, dtype_id, size_bin, ..., d2_bin].
   *
   * @param features Data statistics (entropy, MAD, d2, size, dtype)
   * @param lib_id Library base id
   * @param preset Compression preset
   * @return 7-element state tuple
   */
  std::array<int, 7> BuildStateTuple(const Features &features,
                                     int lib_id,
                                     ctp::CompressionPreset preset) const;

  /**
   * Bin a continuous feature value using lower_bound on bin_edges.
   *
   * @param value Feature value (entropy, MAD, d2, size)
   * @param edges Bin edge array for this feature
   * @return Bin index [0, edges.size()]
   */
  int BinFeature(double value, const std::vector<float> &edges) const;
  /** Fold a (count-weighted) value into the codec's marginal mean. */
  void AccumulateMarginal(const std::array<int, 3> &key,
                          const Prediction &value, double count);
  /** Copy the bin edges out of binning_params_ once. */
  void LoadEdges();

  bool size_only_ = false;                ///< HCompress: size bin only
  bool frozen_ = false;                   ///< ignore observations (ablation)
  nlohmann::json qtable_;                 ///< Full qtable.json content
  nlohmann::json binning_params_;         ///< Full binning_params.json content

  std::map<std::string, int> lib_to_id_;  ///< Library name -> base id
  std::map<std::string, int> dtype_to_id_;  ///< dtype name -> id

  /**
   * Online state: sample count and means for updating.
   */
  /** Table entry: running means and the number of samples behind them. */
  struct Entry {
    Prediction mean_;        ///< Mean compress/decompress time and ratio
    double count_ = 0.0;     ///< Samples (training sample_count, then online)
  };
  std::array<std::vector<float>, 4> edges_;  ///< size, entropy, mad, d2 edges
  /** Per (library_id, config_id, datatype_id) sample-weighted mean over all
   *  bins; the fallback for a state the table has never seen. */
  std::map<std::array<int, 3>, Entry> marginal_;

  /**
   * Hash function for state tuples (used in unordered_map).
   * Packs 7 ints into a uint64 for fast hashing.
   */
  struct StateHash {
    size_t operator()(const std::array<int, 7> &state) const {
      // Pack the 7 integers into a single uint64
      // Each int fits in 9 bits: max value is 255 (< 512)
      uint64_t hash = 0;
      hash |= (static_cast<uint64_t>(state[0] & 0xFF) << 0);
      hash |= (static_cast<uint64_t>(state[1] & 0xFF) << 8);
      hash |= (static_cast<uint64_t>(state[2] & 0xFF) << 16);
      hash |= (static_cast<uint64_t>(state[3] & 0xFF) << 24);
      hash |= (static_cast<uint64_t>(state[4] & 0xFF) << 32);
      hash |= (static_cast<uint64_t>(state[5] & 0xFF) << 40);
      hash |= (static_cast<uint64_t>(state[6] & 0xFF) << 48);
      return hash;
    }
  };

  /**
   * Memoized state->values lookup for O(1) predictions.
   * Built at Load time from qtable_["states"].
   */
  std::unordered_map<std::array<int, 7>, Entry, StateHash> state_cache_;
};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_QTABLE_PREDICTOR_H_
