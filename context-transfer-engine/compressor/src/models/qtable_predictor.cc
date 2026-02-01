/**
 * @file qtable_predictor.cc
 * @brief Implementation of Q-Table compression predictor
 */

#include "wrp_cte/compressor/models/qtable_predictor.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace wrp_cte::compressor {

// ============================================================================
// QState implementation
// ============================================================================

std::string QState::ToString() const {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < bins.size(); ++i) {
    if (i > 0) ss << ",";
    ss << bins[i];
  }
  ss << "]";
  return ss.str();
}

// ============================================================================
// QTablePredictor implementation
// ============================================================================

QTablePredictor::QTablePredictor()
    : config_(), table_ready_(false), unknown_count_(0) {}

QTablePredictor::QTablePredictor(const QTableConfig& config)
    : config_(config), table_ready_(false), unknown_count_(0) {}

QTablePredictor::~QTablePredictor() = default;

QTablePredictor::QTablePredictor(QTablePredictor&& other) noexcept
    : config_(other.config_),
      qtable_(std::move(other.qtable_)),
      bin_edges_(std::move(other.bin_edges_)),
      global_average_(other.global_average_),
      table_ready_(other.table_ready_),
      unknown_count_(other.unknown_count_) {
  other.table_ready_ = false;
}

QTablePredictor& QTablePredictor::operator=(QTablePredictor&& other) noexcept {
  if (this != &other) {
    config_ = other.config_;
    qtable_ = std::move(other.qtable_);
    bin_edges_ = std::move(other.bin_edges_);
    global_average_ = other.global_average_;
    table_ready_ = other.table_ready_;
    unknown_count_ = other.unknown_count_;
    other.table_ready_ = false;
  }
  return *this;
}

bool QTablePredictor::Load(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Load binning parameters
    std::string binning_path = model_dir + "/binning_params.json";
    std::ifstream binning_file(binning_path);
    if (!binning_file.is_open()) {
      return false;
    }

    json binning_json;
    binning_file >> binning_json;
    binning_file.close();

    config_.n_bins = binning_json["n_bins"];
    config_.use_nearest_neighbor = binning_json.value("use_nearest_neighbor", false);
    config_.nn_k = binning_json.value("nn_k", 5);

    // Load bin edges
    bin_edges_.clear();
    for (const auto& edges : binning_json["bin_edges"]) {
      std::vector<float> edge_vec = edges.get<std::vector<float>>();
      bin_edges_.push_back(edge_vec);
    }

    // Load Q-table
    std::string qtable_path = model_dir + "/qtable.json";
    std::ifstream qtable_file(qtable_path);
    if (!qtable_file.is_open()) {
      return false;
    }

    json qtable_json;
    qtable_file >> qtable_json;
    qtable_file.close();

    qtable_.clear();
    for (const auto& entry : qtable_json["states"]) {
      QState state;
      state.bins.fill(0);  // Initialize all bins to 0
      auto bins_array = entry["state"].get<std::vector<int>>();
      // Copy only the number of bins present in the JSON (typically 6 from Python)
      size_t n_bins_to_copy = std::min(bins_array.size(), state.bins.size());
      std::copy_n(bins_array.begin(), n_bins_to_copy, state.bins.begin());

      QValue value(
        entry["compression_ratio"],
        entry["psnr_db"],
        entry["compression_time_ms"],
        entry["sample_count"]
      );

      qtable_[state] = value;
    }

    // Load global average
    if (qtable_json.contains("global_average")) {
      const auto& ga = qtable_json["global_average"];
      global_average_ = QValue(
        ga["compression_ratio"],
        ga["psnr_db"],
        ga["compression_time_ms"],
        ga["sample_count"]
      );
    }

    table_ready_ = true;
    return true;

  } catch (const std::exception& e) {
    table_ready_ = false;
    return false;
  }
}

bool QTablePredictor::Save(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!table_ready_) {
    return false;
  }

  try {
    // Create directory if it doesn't exist
    std::string mkdir_cmd = "mkdir -p " + model_dir;
    [[maybe_unused]] int result = system(mkdir_cmd.c_str());

    // Save binning parameters
    json binning_json;
    binning_json["n_bins"] = config_.n_bins;
    binning_json["use_nearest_neighbor"] = config_.use_nearest_neighbor;
    binning_json["nn_k"] = config_.nn_k;
    binning_json["bin_edges"] = bin_edges_;

    std::string binning_path = model_dir + "/binning_params.json";
    std::ofstream binning_file(binning_path);
    if (!binning_file.is_open()) {
      return false;
    }
    binning_file << binning_json.dump(2);
    binning_file.close();

    // Save Q-table
    json qtable_json;
    qtable_json["num_states"] = qtable_.size();

    std::vector<json> states_array;
    for (const auto& [state, value] : qtable_) {
      json entry;
      entry["state"] = std::vector<int>(state.bins.begin(), state.bins.end());
      entry["compression_ratio"] = value.compression_ratio;
      entry["psnr_db"] = value.psnr_db;
      entry["compression_time_ms"] = value.compression_time_ms;
      entry["sample_count"] = value.sample_count;
      states_array.push_back(entry);
    }
    qtable_json["states"] = states_array;

    // Save global average
    json ga_json;
    ga_json["compression_ratio"] = global_average_.compression_ratio;
    ga_json["psnr_db"] = global_average_.psnr_db;
    ga_json["compression_time_ms"] = global_average_.compression_time_ms;
    ga_json["sample_count"] = global_average_.sample_count;
    qtable_json["global_average"] = ga_json;

    std::string qtable_path = model_dir + "/qtable.json";
    std::ofstream qtable_file(qtable_path);
    if (!qtable_file.is_open()) {
      return false;
    }
    qtable_file << qtable_json.dump(2);
    qtable_file.close();

    return true;

  } catch (const std::exception& e) {
    return false;
  }
}

bool QTablePredictor::IsReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return table_ready_;
}

CompressionPrediction QTablePredictor::Predict(const CompressionFeatures& features) {
  auto start = std::chrono::high_resolution_clock::now();

  QState state = DiscretizeFeatures(features);
  QValue value = GetPrediction(state);

  auto end = std::chrono::high_resolution_clock::now();
  double inference_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

  return CompressionPrediction(
    value.compression_ratio,
    value.psnr_db,
    value.compression_time_ms,
    inference_time_ms
  );
}

std::vector<CompressionPrediction> QTablePredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());

  unknown_count_ = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto& features : batch) {
    QState state = DiscretizeFeatures(features);
    QValue value = GetPrediction(state);

    results.emplace_back(value.compression_ratio, value.psnr_db,
                         value.compression_time_ms, 0.0);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double per_sample_time = batch.empty() ? 0.0 : total_time_ms / batch.size();

  // Update inference time for all predictions
  for (auto& result : results) {
    result.inference_time_ms = per_sample_time;
  }

  return results;
}

bool QTablePredictor::Train(const std::vector<CompressionFeatures>& features,
                             const std::vector<TrainingLabels>& labels) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  try {
    // Step 1: Build binning edges from training data
    BuildBinningEdges(features);

    // Step 2: Discretize all training samples and build Q-table
    qtable_.clear();
    double total_ratio = 0.0;
    double total_psnr = 0.0;
    double total_time = 0.0;

    for (size_t i = 0; i < features.size(); ++i) {
      QState state = DiscretizeFeatures(features[i]);

      // Update Q-table entry (accumulate for averaging)
      auto& qvalue = qtable_[state];
      double n = static_cast<double>(qvalue.sample_count);
      qvalue.compression_ratio = static_cast<float>(
        (qvalue.compression_ratio * n + labels[i].compression_ratio) / (n + 1));
      qvalue.psnr_db = static_cast<float>(
        (qvalue.psnr_db * n + labels[i].psnr_db) / (n + 1));
      qvalue.compression_time_ms = static_cast<float>(
        (qvalue.compression_time_ms * n + labels[i].compression_time_ms) / (n + 1));
      qvalue.sample_count++;

      // Accumulate for global average
      total_ratio += labels[i].compression_ratio;
      total_psnr += labels[i].psnr_db;
      total_time += labels[i].compression_time_ms;
    }

    // Compute global average
    size_t n = features.size();
    global_average_ = QValue(
      static_cast<float>(total_ratio / n),
      static_cast<float>(total_psnr / n),
      static_cast<float>(total_time / n),
      n
    );

    table_ready_ = true;
    return true;

  } catch (const std::exception& e) {
    table_ready_ = false;
    return false;
  }
}

std::string QTablePredictor::GetStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream ss;
  ss << "Q-Table Statistics:\n";
  ss << "  States: " << qtable_.size() << "\n";
  ss << "  Bins: " << config_.n_bins << "\n";
  ss << "  Use NN: " << (config_.use_nearest_neighbor ? "Yes" : "No") << "\n";
  if (config_.use_nearest_neighbor) {
    ss << "  NN k: " << config_.nn_k << "\n";
  }
  ss << "  Global Avg Ratio: " << global_average_.compression_ratio << "\n";
  ss << "  Ready: " << (table_ready_ ? "Yes" : "No") << "\n";
  return ss.str();
}

// ============================================================================
// Private methods
// ============================================================================

QState QTablePredictor::DiscretizeFeatures(const CompressionFeatures& features) const {
  QState state;

  // Match Python feature order:
  // 0: library_id (from library_config_id) - discrete
  // 1: config_id (from config_* one-hot) - discrete
  // 2: datatype_id (from data_type_* one-hot) - discrete
  // 3: data_size (chunk_size_bytes) - continuous
  // 4: shannon_entropy - continuous
  // 5: mean_absolute_deviation (mad) - continuous

  // Feature 0: library_id (discrete, just cast)
  state.bins[0] = static_cast<int>(features.library_config_id);

  // Feature 1: config_id (decode from one-hot encoding)
  if (features.config_fast > 0.5) {
    state.bins[1] = 3;  // fast
  } else if (features.config_best > 0.5) {
    state.bins[1] = 1;  // best
  } else if (features.config_balanced > 0.5) {
    state.bins[1] = 0;  // balanced
  } else {
    state.bins[1] = 2;  // default
  }

  // Feature 2: datatype_id (decode from one-hot encoding)
  if (features.data_type_char > 0.5) {
    state.bins[2] = 0;  // char
  } else if (features.data_type_float > 0.5) {
    state.bins[2] = 1;  // float
  } else {
    state.bins[2] = 2;  // int
  }

  // Features 3-5: continuous features (bin using edges)
  std::array<double, 3> continuous_values = {
    features.chunk_size_bytes,
    features.shannon_entropy,
    features.mad
  };

  for (size_t i = 0; i < 3; ++i) {
    size_t feature_idx = i + 3;  // Features 3, 4, 5
    if (feature_idx < bin_edges_.size() && !bin_edges_[feature_idx].empty()) {
      const auto& edges = bin_edges_[feature_idx];
      float value = static_cast<float>(continuous_values[i]);

      // Use lower_bound to find the bin
      auto it = std::lower_bound(edges.begin(), edges.end(), value);
      int bin_idx = static_cast<int>(std::distance(edges.begin(), it));
      bin_idx = std::min(bin_idx, config_.n_bins - 1);
      bin_idx = std::max(bin_idx, 0);
      state.bins[feature_idx] = bin_idx;
    } else {
      state.bins[feature_idx] = 0;
    }
  }

  // Features 6-10 are not used in Python model (set to 0)
  for (size_t i = 6; i < state.bins.size(); ++i) {
    state.bins[i] = 0;
  }

  return state;
}

void QTablePredictor::BuildBinningEdges(const std::vector<CompressionFeatures>& features) {
  bin_edges_.clear();
  bin_edges_.resize(11);  // 11 features

  // Skip binning for categorical features (already discrete)
  // Bin continuous features (chunk_size, target_cpu_util, shannon_entropy, mad, second_derivative)
  std::vector<size_t> continuous_indices = {0, 1, 2, 3, 4};  // Indices in ToVector()

  for (size_t idx : continuous_indices) {
    // Collect all values for this feature
    std::vector<float> values;
    values.reserve(features.size());
    for (const auto& f : features) {
      auto vec = f.ToVector();
      values.push_back(vec[idx]);
    }

    // Sort values
    std::sort(values.begin(), values.end());

    // Compute percentile-based bin edges
    std::vector<float> edges;
    for (int i = 1; i < config_.n_bins; ++i) {
      double percentile = static_cast<double>(i) / config_.n_bins;
      size_t pos = static_cast<size_t>(percentile * values.size());
      pos = std::min(pos, values.size() - 1);
      edges.push_back(values[pos]);
    }

    // Remove duplicates and ensure edges are unique
    auto last = std::unique(edges.begin(), edges.end());
    edges.erase(last, edges.end());

    bin_edges_[idx] = edges;
  }
}

std::vector<QState> QTablePredictor::FindNearestNeighbors(const QState& state, int k) const {
  std::vector<std::pair<double, QState>> distances;

  for (const auto& [known_state, _] : qtable_) {
    double dist = ComputeDistance(state, known_state);
    distances.emplace_back(dist, known_state);
  }

  // Partial sort to get k nearest
  size_t k_clamped = std::min(static_cast<size_t>(k), distances.size());
  std::partial_sort(distances.begin(),
                   distances.begin() + k_clamped,
                   distances.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });

  std::vector<QState> neighbors;
  neighbors.reserve(k_clamped);
  for (size_t i = 0; i < k_clamped; ++i) {
    neighbors.push_back(distances[i].second);
  }

  return neighbors;
}

double QTablePredictor::ComputeDistance(const QState& s1, const QState& s2) const {
  double dist = 0.0;
  for (size_t i = 0; i < s1.bins.size(); ++i) {
    double diff = static_cast<double>(s1.bins[i] - s2.bins[i]);
    dist += diff * diff;
  }
  return std::sqrt(dist);
}

QValue QTablePredictor::GetPrediction(const QState& state) {
  auto it = qtable_.find(state);

  if (it != qtable_.end()) {
    // State found in Q-table
    return it->second;
  }

  // Unknown state
  unknown_count_++;

  if (config_.use_nearest_neighbor && !qtable_.empty()) {
    // Use nearest neighbor fallback
    auto neighbors = FindNearestNeighbors(state, config_.nn_k);

    if (!neighbors.empty()) {
      // Average predictions from neighbors
      float avg_ratio = 0.0f;
      float avg_psnr = 0.0f;
      float avg_time = 0.0f;

      for (const auto& neighbor : neighbors) {
        const QValue& nvalue = qtable_[neighbor];
        avg_ratio += nvalue.compression_ratio;
        avg_psnr += nvalue.psnr_db;
        avg_time += nvalue.compression_time_ms;
      }

      size_t n = neighbors.size();
      return QValue(avg_ratio / n, avg_psnr / n, avg_time / n, n);
    }
  }

  // Fallback to global average
  return global_average_;
}

}  // namespace wrp_cte::compressor
