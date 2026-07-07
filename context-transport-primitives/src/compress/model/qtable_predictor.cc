/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file qtable_predictor.cc
 * @brief Implementation of Q-Table compression predictor.
 */

#include "clio_ctp/compress/model/qtable_predictor.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>

namespace ctp::compress::model {

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

QTablePredictor& QTablePredictor::operator=(
    QTablePredictor&& other) noexcept {
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
    // Load binning parameters from qtable.csv
    std::string table_path = model_dir + "/qtable.csv";
    std::ifstream table_file(table_path);
    if (!table_file.is_open()) {
      return false;
    }

    std::string line;

    // Parse config section
    while (std::getline(table_file, line)) {
      if (line.empty() || line[0] == '#') continue;
      if (line.find("global_avg") != std::string::npos) break;

      size_t eq_pos = line.find('=');
      if (eq_pos != std::string::npos) {
        std::string key = line.substr(0, eq_pos);
        std::string val = line.substr(eq_pos + 1);

        if (key == "n_bins") {
          config_.n_bins = std::stoi(val);
        } else if (key == "use_nearest_neighbor") {
          config_.use_nearest_neighbor = (std::stoi(val) != 0);
        } else if (key == "nn_k") {
          config_.nn_k = std::stoi(val);
        }
      }
    }

    // Parse global average line
    if (!line.empty() && line.find("global_avg") != std::string::npos) {
      std::getline(table_file, line);
      std::istringstream iss(line);
      float ga_ratio, ga_psnr, ga_time;
      size_t ga_count;
      char comma;
      iss >> ga_ratio >> comma >> ga_psnr >> comma >> ga_time >> comma
          >> ga_count;
      global_average_ =
          QValue(ga_ratio, ga_psnr, ga_time, ga_count);
    }

    // Skip to Q-table section
    while (std::getline(table_file, line)) {
      if (line.find("# Q-Table States") != std::string::npos) break;
    }

    // Parse Q-table entries
    qtable_.clear();
    while (std::getline(table_file, line)) {
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      QState state;
      for (size_t i = 0; i < state.bins.size(); ++i) {
        int bin_val;
        char comma;
        iss >> bin_val;
        state.bins[i] = bin_val;
        if (i < state.bins.size() - 1) iss >> comma;
      }

      float ratio, psnr, time;
      size_t count;
      char comma;
      iss >> comma >> ratio >> comma >> psnr >> comma >> time >> comma
          >> count;

      qtable_[state] = QValue(ratio, psnr, time, count);
    }

    // Load binning edges
    std::string binning_path = model_dir + "/binning.csv";
    std::ifstream binning_file(binning_path);
    bin_edges_.clear();
    bin_edges_.resize(11);

    size_t feature_idx = 0;
    while (feature_idx < bin_edges_.size() &&
           std::getline(binning_file, line)) {
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      float edge_val;
      while (iss >> edge_val) {
        bin_edges_[feature_idx].push_back(edge_val);
        char comma;
        iss >> comma;
      }
      feature_idx++;
    }

    binning_file.close();
    table_file.close();

    table_ready_ = true;
    return true;

  } catch (const std::exception&) {
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
    // Create directory with system call
    std::string mkdir_cmd = "mkdir -p " + model_dir;
    [[maybe_unused]] int result = system(mkdir_cmd.c_str());

    // Save Q-table and config to qtable.csv
    std::string table_path = model_dir + "/qtable.csv";
    std::ofstream table_file(table_path);
    if (!table_file.is_open()) {
      return false;
    }

    // Write config section
    table_file << "# Q-Table Configuration\n";
    table_file << "n_bins=" << config_.n_bins << "\n";
    table_file << "use_nearest_neighbor="
               << (config_.use_nearest_neighbor ? 1 : 0) << "\n";
    table_file << "nn_k=" << config_.nn_k << "\n";

    // Write global average
    table_file << "global_avg_ratio,global_avg_psnr,global_avg_time,"
                  "global_avg_count\n";
    table_file << global_average_.compression_ratio << ","
               << global_average_.psnr_db << ","
               << global_average_.compression_time_ms << ","
               << global_average_.sample_count << "\n";

    // Write Q-table header and entries
    table_file << "# Q-Table States\n";
    table_file << "bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,"
                  "bin10,ratio,psnr,time,count\n";

    for (const auto& [state, value] : qtable_) {
      for (size_t i = 0; i < state.bins.size(); ++i) {
        if (i > 0) table_file << ",";
        table_file << state.bins[i];
      }
      table_file << "," << value.compression_ratio << ","
                 << value.psnr_db << "," << value.compression_time_ms << ","
                 << value.sample_count << "\n";
    }

    table_file.close();

    // Save binning edges to binning.csv
    std::string binning_path = model_dir + "/binning.csv";
    std::ofstream binning_file(binning_path);
    if (!binning_file.is_open()) {
      return false;
    }

    binning_file << "# Binning Edges (feature 0 to 10)\n";
    for (size_t i = 0; i < bin_edges_.size(); ++i) {
      for (size_t j = 0; j < bin_edges_[i].size(); ++j) {
        if (j > 0) binning_file << ",";
        binning_file << bin_edges_[i][j];
      }
      binning_file << "\n";
    }

    binning_file.close();
    return true;

  } catch (const std::exception&) {
    return false;
  }
}

bool QTablePredictor::IsReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return table_ready_;
}

CompressionPrediction QTablePredictor::Predict(
    const CompressionFeatures& features) {
  auto start = std::chrono::high_resolution_clock::now();

  QState state = DiscretizeFeatures(features);
  QValue value = GetPrediction(state);

  auto end = std::chrono::high_resolution_clock::now();
  double inference_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return CompressionPrediction(value.compression_ratio, value.psnr_db,
                               value.compression_time_ms, inference_time_ms);
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
  double total_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double per_sample_time = batch.empty() ? 0.0 : total_time_ms / batch.size();

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
    BuildBinningEdges(features);

    qtable_.clear();
    double total_ratio = 0.0;
    double total_psnr = 0.0;
    double total_time = 0.0;

    for (size_t i = 0; i < features.size(); ++i) {
      QState state = DiscretizeFeatures(features[i]);

      auto& qvalue = qtable_[state];
      double n = static_cast<double>(qvalue.sample_count);
      qvalue.compression_ratio = static_cast<float>(
          (qvalue.compression_ratio * n + labels[i].compression_ratio) /
          (n + 1));
      qvalue.psnr_db = static_cast<float>(
          (qvalue.psnr_db * n + labels[i].psnr_db) / (n + 1));
      qvalue.compression_time_ms = static_cast<float>(
          (qvalue.compression_time_ms * n + labels[i].compression_time_ms) /
          (n + 1));
      qvalue.sample_count++;

      total_ratio += labels[i].compression_ratio;
      total_psnr += labels[i].psnr_db;
      total_time += labels[i].compression_time_ms;
    }

    size_t n = features.size();
    global_average_ = QValue(
        static_cast<float>(total_ratio / n),
        static_cast<float>(total_psnr / n),
        static_cast<float>(total_time / n), n);

    table_ready_ = true;
    return true;

  } catch (const std::exception&) {
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
  ss << "  Use NN: " << (config_.use_nearest_neighbor ? "Yes" : "No")
     << "\n";
  if (config_.use_nearest_neighbor) {
    ss << "  NN k: " << config_.nn_k << "\n";
  }
  ss << "  Global Avg Ratio: " << global_average_.compression_ratio << "\n";
  ss << "  Ready: " << (table_ready_ ? "Yes" : "No") << "\n";
  return ss.str();
}

QState QTablePredictor::DiscretizeFeatures(
    const CompressionFeatures& features) const {
  QState state;

  state.bins[0] = static_cast<int>(features.library_config_id);

  if (features.config_fast > 0.5) {
    state.bins[1] = 3;
  } else if (features.config_best > 0.5) {
    state.bins[1] = 1;
  } else if (features.config_balanced > 0.5) {
    state.bins[1] = 0;
  } else {
    state.bins[1] = 2;
  }

  if (features.data_type_char > 0.5) {
    state.bins[2] = 0;
  } else if (features.data_type_float > 0.5) {
    state.bins[2] = 1;
  } else {
    state.bins[2] = 2;
  }

  std::array<double, 3> continuous_values = {
      features.chunk_size_bytes, features.shannon_entropy, features.mad};

  for (size_t i = 0; i < 3; ++i) {
    size_t feature_idx = i + 3;
    if (feature_idx < bin_edges_.size() && !bin_edges_[feature_idx].empty()) {
      const auto& edges = bin_edges_[feature_idx];
      float value = static_cast<float>(continuous_values[i]);

      auto it = std::lower_bound(edges.begin(), edges.end(), value);
      int bin_idx = static_cast<int>(std::distance(edges.begin(), it));
      bin_idx = std::min(bin_idx, config_.n_bins - 1);
      bin_idx = std::max(bin_idx, 0);
      state.bins[feature_idx] = bin_idx;
    } else {
      state.bins[feature_idx] = 0;
    }
  }

  for (size_t i = 6; i < state.bins.size(); ++i) {
    state.bins[i] = 0;
  }

  return state;
}

void QTablePredictor::BuildBinningEdges(
    const std::vector<CompressionFeatures>& features) {
  bin_edges_.clear();
  bin_edges_.resize(11);

  std::vector<size_t> continuous_indices = {0, 1, 2, 3, 4};

  for (size_t idx : continuous_indices) {
    std::vector<float> values;
    values.reserve(features.size());
    for (const auto& f : features) {
      auto vec = f.ToVector();
      values.push_back(vec[idx]);
    }

    std::sort(values.begin(), values.end());

    std::vector<float> edges;
    for (int i = 1; i < config_.n_bins; ++i) {
      double percentile = static_cast<double>(i) / config_.n_bins;
      size_t pos = static_cast<size_t>(percentile * values.size());
      pos = std::min(pos, values.size() - 1);
      edges.push_back(values[pos]);
    }

    auto last = std::unique(edges.begin(), edges.end());
    edges.erase(last, edges.end());

    bin_edges_[idx] = edges;
  }
}

std::vector<QState> QTablePredictor::FindNearestNeighbors(
    const QState& state, int k) const {
  std::vector<std::pair<double, QState>> distances;

  for (const auto& [known_state, _] : qtable_) {
    double dist = ComputeDistance(state, known_state);
    distances.emplace_back(dist, known_state);
  }

  size_t k_clamped = std::min(static_cast<size_t>(k), distances.size());
  std::partial_sort(distances.begin(), distances.begin() + k_clamped,
                    distances.end(),
                    [](const auto& a, const auto& b) {
                      return a.first < b.first;
                    });

  std::vector<QState> neighbors;
  neighbors.reserve(k_clamped);
  for (size_t i = 0; i < k_clamped; ++i) {
    neighbors.push_back(distances[i].second);
  }

  return neighbors;
}

double QTablePredictor::ComputeDistance(const QState& s1,
                                        const QState& s2) const {
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
    return it->second;
  }

  unknown_count_++;

  if (config_.use_nearest_neighbor && !qtable_.empty()) {
    auto neighbors = FindNearestNeighbors(state, config_.nn_k);

    if (!neighbors.empty()) {
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

  return global_average_;
}

}  // namespace ctp::compress::model
