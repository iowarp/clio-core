/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file linreg_table_predictor.cc
 * @brief Implementation of linear regression table predictor.
 */

#include "clio_ctp/compress/model/linreg_table_predictor.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>

namespace ctp::compress::model {

LinRegTablePredictor::LinRegTablePredictor()
    : config_(), ready_(false) {
  std::vector<std::string> libs = {"BZIP2", "ZSTD", "LZ4", "ZLIB",
                                    "LZMA", "BROTLI", "SNAPPY", "Blosc2"};
  std::vector<std::string> presets = {"fast", "balanced", "best", "default"};

  for (size_t i = 0; i < libs.size(); ++i) {
    int base_id = static_cast<int>(i + 1);
    for (size_t j = 0; j < presets.size(); ++j) {
      int preset_id = static_cast<int>(j);
      int lib_config_id = base_id * 10 + preset_id;
      id_to_lib_config_[lib_config_id] = {libs[i], presets[j]};
    }
  }
}

LinRegTablePredictor::LinRegTablePredictor(const LinRegTableConfig& config)
    : config_(config), ready_(false) {
  std::vector<std::string> libs = {"BZIP2", "ZSTD", "LZ4", "ZLIB",
                                    "LZMA", "BROTLI", "SNAPPY", "Blosc2"};
  std::vector<std::string> presets = {"fast", "balanced", "best", "default"};

  for (size_t i = 0; i < libs.size(); ++i) {
    int base_id = static_cast<int>(i + 1);
    for (size_t j = 0; j < presets.size(); ++j) {
      int preset_id = static_cast<int>(j);
      int lib_config_id = base_id * 10 + preset_id;
      id_to_lib_config_[lib_config_id] = {libs[i], presets[j]};
    }
  }
}

LinRegTablePredictor::~LinRegTablePredictor() = default;

LinRegTablePredictor::LinRegTablePredictor(LinRegTablePredictor&& other) noexcept
    : config_(std::move(other.config_)),
      table_(std::move(other.table_)),
      ready_(other.ready_),
      id_to_lib_config_(std::move(other.id_to_lib_config_)) {
  other.ready_ = false;
}

LinRegTablePredictor& LinRegTablePredictor::operator=(
    LinRegTablePredictor&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = std::move(other.config_);
    table_ = std::move(other.table_);
    ready_ = other.ready_;
    id_to_lib_config_ = std::move(other.id_to_lib_config_);
    other.ready_ = false;
  }
  return *this;
}

bool LinRegTablePredictor::Load(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string table_path = model_dir + "/linreg_table.csv";
  std::ifstream file(table_path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  table_.clear();

  // Parse config section
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      if (line.find("Models") != std::string::npos) break;
      continue;
    }

    size_t eq_pos = line.find('=');
    if (eq_pos != std::string::npos) {
      std::string key = line.substr(0, eq_pos);
      std::string val = line.substr(eq_pos + 1);

      if (key == "min_samples") {
        config_.min_samples = std::stod(val);
      } else if (key == "fallback_compress_time") {
        config_.fallback_compress_time = std::stod(val);
      } else if (key == "fallback_decompress_time") {
        config_.fallback_decompress_time = std::stod(val);
      } else if (key == "fallback_compress_ratio") {
        config_.fallback_compress_ratio = std::stod(val);
      }
    }
  }

  // Skip header line
  std::getline(file, line);

  // Parse model entries
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    LinRegTableKey key;
    LinearRegressionCoeffs coeffs;
    std::string library, config, data_type, distribution;

    char comma;
    if (!(iss >> std::quoted(library) >> comma >> std::quoted(config) >>
          comma >> std::quoted(data_type) >> comma >> std::quoted(distribution)
          >> comma >> coeffs.slope_compress_time >> comma >>
          coeffs.intercept_compress_time >> comma >>
          coeffs.slope_decompress_time >> comma >>
          coeffs.intercept_decompress_time >> comma >>
          coeffs.slope_compress_ratio >> comma >>
          coeffs.intercept_compress_ratio >> comma >> coeffs.sample_count >>
          comma >> coeffs.r2_compress_time >> comma >>
          coeffs.r2_decompress_time >> comma >> coeffs.r2_compress_ratio)) {
      continue;
    }

    key = LinRegTableKey(library, config, data_type, distribution);
    table_[key] = coeffs;
  }

  file.close();

  ready_ = !table_.empty();
  return ready_;
}

bool LinRegTablePredictor::Save(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string mkdir_cmd = "mkdir -p " + model_dir;
  [[maybe_unused]] int result = system(mkdir_cmd.c_str());

  std::string table_path = model_dir + "/linreg_table.csv";
  std::ofstream file(table_path);
  if (!file.is_open()) {
    return false;
  }

  // Write config section
  file << "# Linear Regression Table Configuration\n";
  file << "min_samples=" << config_.min_samples << "\n";
  file << "fallback_compress_time=" << config_.fallback_compress_time << "\n";
  file << "fallback_decompress_time=" << config_.fallback_decompress_time
       << "\n";
  file << "fallback_compress_ratio=" << config_.fallback_compress_ratio
       << "\n";

  // Write header
  file << "# Models\n";
  file << "library,config,data_type,distribution,slope_ct,intercept_ct,"
          "slope_dt,intercept_dt,slope_ratio,intercept_ratio,sample_count,"
          "r2_ct,r2_dt,r2_ratio\n";

  // Write table entries
  for (const auto& [key, coeffs] : table_) {
    file << std::quoted(key.library) << "," << std::quoted(key.config) << ","
         << std::quoted(key.data_type) << ","
         << std::quoted(key.distribution) << ","
         << coeffs.slope_compress_time << ","
         << coeffs.intercept_compress_time << ","
         << coeffs.slope_decompress_time << ","
         << coeffs.intercept_decompress_time << ","
         << coeffs.slope_compress_ratio << ","
         << coeffs.intercept_compress_ratio << "," << coeffs.sample_count
         << "," << coeffs.r2_compress_time << ","
         << coeffs.r2_decompress_time << "," << coeffs.r2_compress_ratio
         << "\n";
  }

  file.close();
  return true;
}

bool LinRegTablePredictor::IsReady() const {
  return ready_;
}

CompressionPrediction LinRegTablePredictor::Predict(
    const CompressionFeatures& features) {
  std::string library, config;
  DecodeLibraryConfigId(static_cast<int>(features.library_config_id),
                        library, config);
  std::string data_type = GetDataTypeFromFeatures(features);
  return PredictByKey(library, config, data_type, features.chunk_size_bytes);
}

std::vector<CompressionPrediction> LinRegTablePredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());
  for (const auto& features : batch) {
    results.push_back(Predict(features));
  }
  return results;
}

CompressionPrediction LinRegTablePredictor::PredictByKey(
    const std::string& library, const std::string& config,
    const std::string& data_type, const std::string& distribution,
    double data_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  CompressionPrediction result;
  LinRegTableKey key(library, config, data_type, distribution);
  auto it = table_.find(key);

  if (it != table_.end()) {
    const auto& coeffs = it->second;
    result.compression_time_ms = coeffs.PredictCompressTime(data_size);
    result.compression_ratio = coeffs.PredictCompressRatio(data_size);
    result.psnr_db = 0.0;
    result.inference_time_ms = 0.001;
  } else {
    result.compression_time_ms = config_.fallback_compress_time;
    result.compression_ratio = config_.fallback_compress_ratio;
    result.psnr_db = 0.0;
    result.inference_time_ms = 0.001;
  }

  return result;
}

CompressionPrediction LinRegTablePredictor::PredictByKey(
    const std::string& library, const std::string& config,
    const std::string& data_type, double data_size) {
  return PredictByKey(library, config, data_type, "", data_size);
}

const LinearRegressionCoeffs* LinRegTablePredictor::GetCoeffs(
    const std::string& library, const std::string& config,
    const std::string& data_type, const std::string& distribution) const {
  LinRegTableKey key(library, config, data_type, distribution);
  auto it = table_.find(key);
  if (it != table_.end()) {
    return &(it->second);
  }
  return nullptr;
}

void LinRegTablePredictor::FitLinearRegression(const std::vector<double>& x,
                                               const std::vector<double>& y,
                                               double& slope,
                                               double& intercept,
                                               double& r2) {
  size_t n = x.size();
  if (n < 2) {
    slope = 0.0;
    intercept = n > 0 ? y[0] : 0.0;
    r2 = 0.0;
    return;
  }

  double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
  double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;

  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double dx = x[i] - mean_x;
    numerator += dx * (y[i] - mean_y);
    denominator += dx * dx;
  }

  if (std::abs(denominator) < 1e-10) {
    slope = 0.0;
    intercept = mean_y;
    r2 = 0.0;
    return;
  }

  slope = numerator / denominator;
  intercept = mean_y - slope * mean_x;

  double ss_res = 0.0;
  double ss_tot = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double y_pred = slope * x[i] + intercept;
    ss_res += (y[i] - y_pred) * (y[i] - y_pred);
    ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
  }

  r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
}

bool LinRegTablePredictor::Train(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels) {
  if (features.empty() || features.size() != labels.size()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  // Group sample indices by the same key the Predict path looks up
  // (library, config, data_type, distribution=""), so a trained model is
  // immediately queryable through Predict().
  std::map<LinRegTableKey, std::vector<size_t>> groups;
  for (size_t i = 0; i < features.size(); ++i) {
    std::string library, config;
    DecodeLibraryConfigId(static_cast<int>(features[i].library_config_id),
                          library, config);
    std::string data_type = GetDataTypeFromFeatures(features[i]);
    groups[LinRegTableKey(library, config, data_type, "")].push_back(i);
  }

  // Ordinary least squares of chunk size -> {compression_time, ratio} per
  // group. A group needs >= 2 points to define a line; smaller groups are
  // skipped (they fall back to the configured defaults at predict time).
  table_.clear();
  for (const auto& [key, idxs] : groups) {
    if (idxs.size() < 2) continue;
    double n = static_cast<double>(idxs.size());
    double sx = 0, sxx = 0, sy_t = 0, sxy_t = 0, sy_r = 0, sxy_r = 0;
    for (size_t i : idxs) {
      double x = features[i].chunk_size_bytes;
      double yt = labels[i].compression_time_ms;
      double yr = labels[i].compression_ratio;
      sx += x;
      sxx += x * x;
      sy_t += yt;
      sxy_t += x * yt;
      sy_r += yr;
      sxy_r += x * yr;
    }
    double denom = n * sxx - sx * sx;
    LinearRegressionCoeffs c;
    if (std::abs(denom) < 1e-12) {
      // Degenerate (all x equal): flat line at the mean.
      c.slope_compress_time = 0.0;
      c.intercept_compress_time = sy_t / n;
      c.slope_compress_ratio = 0.0;
      c.intercept_compress_ratio = sy_r / n;
    } else {
      c.slope_compress_time = (n * sxy_t - sx * sy_t) / denom;
      c.intercept_compress_time = (sy_t - c.slope_compress_time * sx) / n;
      c.slope_compress_ratio = (n * sxy_r - sx * sy_r) / denom;
      c.intercept_compress_ratio = (sy_r - c.slope_compress_ratio * sx) / n;
    }
    // No separate decompress labels in the unified interface; mirror compress.
    c.slope_decompress_time = c.slope_compress_time;
    c.intercept_decompress_time = c.intercept_compress_time;
    table_[key] = c;
  }

  ready_ = !table_.empty();
  return ready_;
}

bool LinRegTablePredictor::TrainFromRaw(
    const std::vector<std::string>& libraries,
    const std::vector<std::string>& configs,
    const std::vector<std::string>& data_types,
    const std::vector<std::string>& distributions,
    const std::vector<double>& data_sizes,
    const std::vector<double>& compress_times,
    const std::vector<double>& decompress_times,
    const std::vector<double>& compress_ratios) {

  size_t n = libraries.size();
  if (n != configs.size() || n != data_types.size() ||
      n != distributions.size() || n != data_sizes.size() ||
      n != compress_times.size() || n != decompress_times.size() ||
      n != compress_ratios.size()) {
    return false;
  }

  std::map<LinRegTableKey, std::vector<size_t>> groups;
  for (size_t i = 0; i < n; ++i) {
    LinRegTableKey key(libraries[i], configs[i], data_types[i],
                       distributions[i]);
    groups[key].push_back(i);
  }

  table_.clear();
  for (const auto& group : groups) {
    const auto& key = group.first;
    const auto& indices = group.second;

    if (indices.size() < static_cast<size_t>(config_.min_samples)) {
      continue;
    }

    std::vector<double> x_data, y_compress, y_decompress, y_ratio;
    x_data.reserve(indices.size());
    y_compress.reserve(indices.size());
    y_decompress.reserve(indices.size());
    y_ratio.reserve(indices.size());

    for (size_t idx : indices) {
      x_data.push_back(data_sizes[idx]);
      y_compress.push_back(compress_times[idx]);
      y_decompress.push_back(decompress_times[idx]);
      y_ratio.push_back(compress_ratios[idx]);
    }

    LinearRegressionCoeffs coeffs;
    coeffs.sample_count = indices.size();

    FitLinearRegression(x_data, y_compress, coeffs.slope_compress_time,
                        coeffs.intercept_compress_time,
                        coeffs.r2_compress_time);

    FitLinearRegression(x_data, y_decompress, coeffs.slope_decompress_time,
                        coeffs.intercept_decompress_time,
                        coeffs.r2_decompress_time);

    FitLinearRegression(x_data, y_ratio, coeffs.slope_compress_ratio,
                        coeffs.intercept_compress_ratio,
                        coeffs.r2_compress_ratio);

    table_[key] = coeffs;
  }

  ready_ = !table_.empty();
  return ready_;
}

void LinRegTablePredictor::DecodeLibraryConfigId(
    int library_config_id, std::string& library, std::string& config) const {
  auto it = id_to_lib_config_.find(library_config_id);
  if (it != id_to_lib_config_.end()) {
    library = it->second.first;
    config = it->second.second;
  } else {
    library = "unknown";
    config = "unknown";
  }
}

std::string LinRegTablePredictor::GetDataTypeFromFeatures(
    const CompressionFeatures& features) const {
  if (features.data_type_char > 0.5) {
    return "char";
  } else if (features.data_type_float > 0.5) {
    return "float";
  }
  return "int";
}

std::string LinRegTablePredictor::GetStatistics() const {
  std::stringstream ss;
  ss << "LinRegTablePredictor Statistics:\n";
  ss << "  Ready: " << (ready_ ? "yes" : "no") << "\n";
  ss << "  Number of models: " << table_.size() << "\n";

  if (!table_.empty()) {
    double avg_r2_compress = 0.0;
    double avg_r2_decompress = 0.0;
    double avg_r2_ratio = 0.0;
    size_t total_samples = 0;

    for (const auto& entry : table_) {
      const auto& coeffs = entry.second;
      avg_r2_compress += coeffs.r2_compress_time;
      avg_r2_decompress += coeffs.r2_decompress_time;
      avg_r2_ratio += coeffs.r2_compress_ratio;
      total_samples += coeffs.sample_count;
    }

    avg_r2_compress /= table_.size();
    avg_r2_decompress /= table_.size();
    avg_r2_ratio /= table_.size();

    ss << "  Total training samples: " << total_samples << "\n";
    ss << "  Average R² (compress_time): " << avg_r2_compress << "\n";
    ss << "  Average R² (decompress_time): " << avg_r2_decompress << "\n";
    ss << "  Average R² (compress_ratio): " << avg_r2_ratio << "\n";
  }

  return ss.str();
}

std::vector<std::string> LinRegTablePredictor::GetLibraries() const {
  std::set<std::string> libs;
  for (const auto& entry : table_) {
    libs.insert(entry.first.library);
  }
  return std::vector<std::string>(libs.begin(), libs.end());
}

std::vector<std::string> LinRegTablePredictor::GetConfigurations() const {
  std::set<std::string> configs;
  for (const auto& entry : table_) {
    configs.insert(entry.first.config);
  }
  return std::vector<std::string>(configs.begin(), configs.end());
}

std::vector<std::string> LinRegTablePredictor::GetDataTypes() const {
  std::set<std::string> types;
  for (const auto& entry : table_) {
    types.insert(entry.first.data_type);
  }
  return std::vector<std::string>(types.begin(), types.end());
}

std::vector<std::string> LinRegTablePredictor::GetDistributions() const {
  std::set<std::string> dists;
  for (const auto& entry : table_) {
    if (!entry.first.distribution.empty()) {
      dists.insert(entry.first.distribution);
    }
  }
  return std::vector<std::string>(dists.begin(), dists.end());
}

}  // namespace ctp::compress::model
