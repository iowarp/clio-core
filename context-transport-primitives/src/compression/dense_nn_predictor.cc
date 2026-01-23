/**
 * @file dense_nn_predictor.cc
 * @brief Implementation of Dense Neural Network compression predictor using MiniDNN
 */

#include "context-transport-primitives/compression/dense_nn_predictor.h"
#include <fstream>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <nlohmann/json.hpp>

namespace hshm {
namespace compress {

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

// ============================================================================
// StandardScaler Implementation
// ============================================================================

void StandardScaler::Fit(const std::vector<std::vector<float>>& data) {
  if (data.empty() || data[0].empty()) {
    return;
  }

  size_t num_features = data[0].size();
  mean_.resize(num_features, 0.0f);
  std_.resize(num_features, 0.0f);

  // Calculate means
  for (const auto& row : data) {
    for (size_t i = 0; i < num_features; ++i) {
      mean_[i] += row[i];
    }
  }
  for (size_t i = 0; i < num_features; ++i) {
    mean_[i] /= static_cast<float>(data.size());
  }

  // Calculate standard deviations
  for (const auto& row : data) {
    for (size_t i = 0; i < num_features; ++i) {
      float diff = row[i] - mean_[i];
      std_[i] += diff * diff;
    }
  }
  for (size_t i = 0; i < num_features; ++i) {
    std_[i] = std::sqrt(std_[i] / static_cast<float>(data.size()));
    // Prevent division by zero
    if (std_[i] < 1e-7f) {
      std_[i] = 1.0f;
    }
  }

  fitted_ = true;
}

std::vector<std::vector<float>> StandardScaler::Transform(
    const std::vector<std::vector<float>>& data) const {
  if (!fitted_ || data.empty()) {
    return data;
  }

  std::vector<std::vector<float>> result;
  result.reserve(data.size());

  for (const auto& row : data) {
    std::vector<float> normalized(row.size());
    for (size_t i = 0; i < row.size(); ++i) {
      normalized[i] = (row[i] - mean_[i]) / std_[i];
    }
    result.push_back(std::move(normalized));
  }

  return result;
}

std::vector<std::vector<float>> StandardScaler::FitTransform(
    const std::vector<std::vector<float>>& data) {
  Fit(data);
  return Transform(data);
}

bool StandardScaler::Save(const std::string& path) const {
  if (!fitted_) {
    return false;
  }

  try {
    json j;
    j["mean"] = mean_;
    j["std"] = std_;

    std::ofstream file(path);
    if (!file.is_open()) {
      return false;
    }
    file << j.dump(2);
    return true;
  } catch (...) {
    return false;
  }
}

bool StandardScaler::Load(const std::string& path) {
  try {
    std::ifstream file(path);
    if (!file.is_open()) {
      return false;
    }

    json j;
    file >> j;

    mean_ = j["mean"].get<std::vector<float>>();
    std_ = j["std"].get<std::vector<float>>();
    fitted_ = true;

    return true;
  } catch (...) {
    return false;
  }
}

// ============================================================================
// DenseNNPredictor Implementation
// ============================================================================

DenseNNPredictor::DenseNNPredictor()
    : ratio_model_ready_(false),
      psnr_model_ready_(false),
      time_model_ready_(false),
      ratio_dropout_rate_(0.0),
      psnr_dropout_rate_(0.0),
      time_dropout_rate_(0.0),
      epsilon_(0.1),
      exploration_enabled_(false),
      rng_(std::random_device{}()),
      experience_count_(0) {
}

DenseNNPredictor::~DenseNNPredictor() = default;

DenseNNPredictor::DenseNNPredictor(DenseNNPredictor&& other) noexcept
    : ratio_network_(std::move(other.ratio_network_)),
      psnr_network_(std::move(other.psnr_network_)),
      time_network_(std::move(other.time_network_)),
      ratio_scaler_(std::move(other.ratio_scaler_)),
      psnr_scaler_(std::move(other.psnr_scaler_)),
      time_scaler_(std::move(other.time_scaler_)),
      ratio_model_ready_(other.ratio_model_ready_),
      psnr_model_ready_(other.psnr_model_ready_),
      time_model_ready_(other.time_model_ready_),
      ratio_hidden_layers_(std::move(other.ratio_hidden_layers_)),
      ratio_dropout_rate_(other.ratio_dropout_rate_),
      psnr_hidden_layers_(std::move(other.psnr_hidden_layers_)),
      psnr_dropout_rate_(other.psnr_dropout_rate_),
      time_hidden_layers_(std::move(other.time_hidden_layers_)),
      time_dropout_rate_(other.time_dropout_rate_),
      experience_buffer_(std::move(other.experience_buffer_)),
      epsilon_(other.epsilon_),
      exploration_enabled_(other.exploration_enabled_),
      rng_(std::move(other.rng_)),
      experience_count_(other.experience_count_) {
  other.ratio_model_ready_ = false;
  other.psnr_model_ready_ = false;
  other.time_model_ready_ = false;
  other.experience_count_ = 0;
}

DenseNNPredictor& DenseNNPredictor::operator=(DenseNNPredictor&& other) noexcept {
  if (this != &other) {
    std::lock_guard<std::mutex> lock(mutex_);
    ratio_network_ = std::move(other.ratio_network_);
    psnr_network_ = std::move(other.psnr_network_);
    time_network_ = std::move(other.time_network_);
    ratio_scaler_ = std::move(other.ratio_scaler_);
    psnr_scaler_ = std::move(other.psnr_scaler_);
    time_scaler_ = std::move(other.time_scaler_);
    ratio_model_ready_ = other.ratio_model_ready_;
    psnr_model_ready_ = other.psnr_model_ready_;
    time_model_ready_ = other.time_model_ready_;
    ratio_hidden_layers_ = std::move(other.ratio_hidden_layers_);
    ratio_dropout_rate_ = other.ratio_dropout_rate_;
    psnr_hidden_layers_ = std::move(other.psnr_hidden_layers_);
    psnr_dropout_rate_ = other.psnr_dropout_rate_;
    time_hidden_layers_ = std::move(other.time_hidden_layers_);
    time_dropout_rate_ = other.time_dropout_rate_;
    experience_buffer_ = std::move(other.experience_buffer_);
    epsilon_ = other.epsilon_;
    exploration_enabled_ = other.exploration_enabled_;
    rng_ = std::move(other.rng_);
    experience_count_ = other.experience_count_;
    other.ratio_model_ready_ = false;
    other.psnr_model_ready_ = false;
    other.time_model_ready_ = false;
    other.experience_count_ = 0;
  }
  return *this;
}

bool DenseNNPredictor::Load(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  ratio_model_ready_ = false;
  psnr_model_ready_ = false;
  time_model_ready_ = false;

  fs::path dir(model_dir);

  // Load compression ratio model
  fs::path ratio_model_path = dir / "compression_ratio_model.bin";
  fs::path ratio_scaler_path = dir / "ratio_scaler.json";
  fs::path ratio_arch_path = dir / "compression_ratio_arch.json";

  if (fs::exists(ratio_model_path) && fs::exists(ratio_scaler_path) &&
      fs::exists(ratio_arch_path)) {
    try {
      ratio_network_ = std::make_unique<MiniDNN::Network>();

      // Load scaler first
      if (!ratio_scaler_.Load(ratio_scaler_path.string())) {
        return false;
      }

      // Load architecture and build network
      std::ifstream arch_file(ratio_arch_path.string());
      if (!arch_file.is_open()) {
        return false;
      }
      json arch_json;
      arch_file >> arch_json;

      std::vector<int> hidden_layers = arch_json["hidden_layers"].get<std::vector<int>>();
      double dropout_rate = arch_json["dropout_rate"].get<double>();
      BuildNetwork(*ratio_network_, hidden_layers, dropout_rate);

      // Initialize network (required before set_parameters)
      ratio_network_->init(0, 0.01, 123);

      // Read model parameters from file
      std::ifstream model_file(ratio_model_path.string(), std::ios::binary);
      if (!model_file.is_open()) {
        return false;
      }

      std::vector<std::vector<double>> params;
      int num_layers;
      model_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

      for (int i = 0; i < num_layers; ++i) {
        int param_size;
        model_file.read(reinterpret_cast<char*>(&param_size), sizeof(param_size));
        std::vector<double> layer_params(param_size);
        model_file.read(reinterpret_cast<char*>(layer_params.data()),
                        param_size * sizeof(double));
        params.push_back(std::move(layer_params));
      }

      ratio_network_->set_parameters(params);
      ratio_model_ready_ = true;
    } catch (...) {
      return false;
    }
  }

  // Load PSNR model (optional)
  fs::path psnr_model_path = dir / "psnr_model.bin";
  fs::path psnr_scaler_path = dir / "psnr_scaler.json";
  fs::path psnr_arch_path = dir / "psnr_arch.json";

  if (fs::exists(psnr_model_path) && fs::exists(psnr_scaler_path) &&
      fs::exists(psnr_arch_path)) {
    try {
      psnr_network_ = std::make_unique<MiniDNN::Network>();

      if (!psnr_scaler_.Load(psnr_scaler_path.string())) {
        return ratio_model_ready_;  // PSNR is optional
      }

      // Load architecture
      std::ifstream arch_file(psnr_arch_path.string());
      if (arch_file.is_open()) {
        json arch_json;
        arch_file >> arch_json;

        std::vector<int> hidden_layers = arch_json["hidden_layers"].get<std::vector<int>>();
        double dropout_rate = arch_json["dropout_rate"].get<double>();
        BuildNetwork(*psnr_network_, hidden_layers, dropout_rate);

        // Initialize network (required before set_parameters)
        psnr_network_->init(0, 0.01, 123);

        // Load parameters
        std::ifstream model_file(psnr_model_path.string(), std::ios::binary);
        if (model_file.is_open()) {
          std::vector<std::vector<double>> params;
          int num_layers;
          model_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

          for (int i = 0; i < num_layers; ++i) {
            int param_size;
            model_file.read(reinterpret_cast<char*>(&param_size), sizeof(param_size));
            std::vector<double> layer_params(param_size);
            model_file.read(reinterpret_cast<char*>(layer_params.data()),
                            param_size * sizeof(double));
            params.push_back(std::move(layer_params));
          }

          psnr_network_->set_parameters(params);
          psnr_model_ready_ = true;
        }
      }
    } catch (...) {
      // PSNR model is optional, continue without it
    }
  }

  // Load compression time model (optional)
  fs::path time_model_path = dir / "compression_time_model.bin";
  fs::path time_scaler_path = dir / "time_scaler.json";
  fs::path time_arch_path = dir / "compression_time_arch.json";

  if (fs::exists(time_model_path) && fs::exists(time_scaler_path) &&
      fs::exists(time_arch_path)) {
    try {
      time_network_ = std::make_unique<MiniDNN::Network>();

      if (!time_scaler_.Load(time_scaler_path.string())) {
        return ratio_model_ready_;  // Time model is optional
      }

      // Load architecture
      std::ifstream arch_file(time_arch_path.string());
      if (arch_file.is_open()) {
        json arch_json;
        arch_file >> arch_json;

        std::vector<int> hidden_layers = arch_json["hidden_layers"].get<std::vector<int>>();
        double dropout_rate = arch_json["dropout_rate"].get<double>();
        BuildNetwork(*time_network_, hidden_layers, dropout_rate);

        // Initialize network (required before set_parameters)
        time_network_->init(0, 0.01, 123);

        // Load parameters
        std::ifstream model_file(time_model_path.string(), std::ios::binary);
        if (model_file.is_open()) {
          std::vector<std::vector<double>> params;
          int num_layers;
          model_file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

          for (int i = 0; i < num_layers; ++i) {
            int param_size;
            model_file.read(reinterpret_cast<char*>(&param_size), sizeof(param_size));
            std::vector<double> layer_params(param_size);
            model_file.read(reinterpret_cast<char*>(layer_params.data()),
                            param_size * sizeof(double));
            params.push_back(std::move(layer_params));
          }

          time_network_->set_parameters(params);
          time_model_ready_ = true;
        }
      }
    } catch (...) {
      // Time model is optional, continue without it
    }
  }

  return ratio_model_ready_;
}

bool DenseNNPredictor::Save(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!ratio_model_ready_) {
    return false;
  }

  fs::path dir(model_dir);
  if (!fs::exists(dir)) {
    fs::create_directories(dir);
  }

  try {
    // Save ratio model
    if (ratio_network_) {
      fs::path ratio_model_path = dir / "compression_ratio_model.bin";
      std::ofstream model_file(ratio_model_path.string(), std::ios::binary);
      if (!model_file.is_open()) {
        return false;
      }

      auto params = ratio_network_->get_parameters();
      int num_layers = static_cast<int>(params.size());
      model_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

      for (const auto& layer_params : params) {
        int param_size = static_cast<int>(layer_params.size());
        model_file.write(reinterpret_cast<const char*>(&param_size), sizeof(param_size));
        model_file.write(reinterpret_cast<const char*>(layer_params.data()),
                         static_cast<std::streamsize>(param_size * sizeof(double)));
      }

      // Save architecture
      fs::path ratio_arch_path = dir / "compression_ratio_arch.json";
      std::ofstream arch_file(ratio_arch_path.string());
      if (!arch_file.is_open()) {
        return false;
      }
      json arch_json;
      arch_json["hidden_layers"] = ratio_hidden_layers_;
      arch_json["dropout_rate"] = ratio_dropout_rate_;
      arch_file << arch_json.dump(2);
    }

    fs::path ratio_scaler_path = dir / "ratio_scaler.json";
    if (!ratio_scaler_.Save(ratio_scaler_path.string())) {
      return false;
    }

    // Save PSNR model if ready
    if (psnr_model_ready_ && psnr_network_) {
      fs::path psnr_model_path = dir / "psnr_model.bin";
      std::ofstream model_file(psnr_model_path.string(), std::ios::binary);
      if (model_file.is_open()) {
        auto params = psnr_network_->get_parameters();
        int num_layers = static_cast<int>(params.size());
        model_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (const auto& layer_params : params) {
          int param_size = static_cast<int>(layer_params.size());
          model_file.write(reinterpret_cast<const char*>(&param_size), sizeof(param_size));
          model_file.write(reinterpret_cast<const char*>(layer_params.data()),
                           static_cast<std::streamsize>(param_size * sizeof(double)));
        }
      }

      // Save architecture
      fs::path psnr_arch_path = dir / "psnr_arch.json";
      std::ofstream arch_file(psnr_arch_path.string());
      if (arch_file.is_open()) {
        json arch_json;
        arch_json["hidden_layers"] = psnr_hidden_layers_;
        arch_json["dropout_rate"] = psnr_dropout_rate_;
        arch_file << arch_json.dump(2);
      }

      fs::path psnr_scaler_path = dir / "psnr_scaler.json";
      psnr_scaler_.Save(psnr_scaler_path.string());
    }

    // Save time model if ready
    if (time_model_ready_ && time_network_) {
      fs::path time_model_path = dir / "compression_time_model.bin";
      std::ofstream model_file(time_model_path.string(), std::ios::binary);
      if (model_file.is_open()) {
        auto params = time_network_->get_parameters();
        int num_layers = static_cast<int>(params.size());
        model_file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (const auto& layer_params : params) {
          int param_size = static_cast<int>(layer_params.size());
          model_file.write(reinterpret_cast<const char*>(&param_size), sizeof(param_size));
          model_file.write(reinterpret_cast<const char*>(layer_params.data()),
                           static_cast<std::streamsize>(param_size * sizeof(double)));
        }
      }

      // Save architecture
      fs::path time_arch_path = dir / "compression_time_arch.json";
      std::ofstream arch_file(time_arch_path.string());
      if (arch_file.is_open()) {
        json arch_json;
        arch_json["hidden_layers"] = time_hidden_layers_;
        arch_json["dropout_rate"] = time_dropout_rate_;
        arch_file << arch_json.dump(2);
      }

      fs::path time_scaler_path = dir / "time_scaler.json";
      time_scaler_.Save(time_scaler_path.string());
    }

    return true;
  } catch (...) {
    return false;
  }
}

bool DenseNNPredictor::IsReady() const {
  return ratio_model_ready_;
}

CompressionPrediction DenseNNPredictor::Predict(const CompressionFeatures& features) {
  std::vector<CompressionFeatures> batch = {features};
  auto results = PredictBatch(batch);
  if (!results.empty()) {
    return results[0];
  }
  return CompressionPrediction();
}

std::vector<CompressionPrediction> DenseNNPredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());

  if (!ratio_model_ready_ || batch.empty() || !ratio_network_) {
    return results;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Convert features to matrix (MiniDNN expects features x samples)
    Eigen::MatrixXd input = FeaturesToMatrix(batch, ratio_scaler_);

    // Predict compression ratios
    Eigen::MatrixXd ratio_output = ratio_network_->predict(input);

    // Predict PSNR if model is ready
    Eigen::MatrixXd psnr_output;
    if (psnr_model_ready_ && psnr_network_) {
      Eigen::MatrixXd psnr_input = FeaturesToMatrix(batch, psnr_scaler_);
      psnr_output = psnr_network_->predict(psnr_input);
    }

    // Predict compression time if model is ready
    Eigen::MatrixXd time_output;
    if (time_model_ready_ && time_network_) {
      Eigen::MatrixXd time_input = FeaturesToMatrix(batch, time_scaler_);
      time_output = time_network_->predict(time_input);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    double per_sample_time_ms = total_time_ms / static_cast<double>(batch.size());

    // Build results
    for (size_t i = 0; i < batch.size(); ++i) {
      double ratio = ratio_output(0, static_cast<Eigen::Index>(i));
      double psnr = 0.0;
      double compress_time = 0.0;
      if (psnr_model_ready_ && psnr_output.cols() > 0 && IsLossyCompression(batch[i])) {
        psnr = psnr_output(0, static_cast<Eigen::Index>(i));
      }
      if (time_model_ready_ && time_output.cols() > 0) {
        compress_time = time_output(0, static_cast<Eigen::Index>(i));
      }
      results.emplace_back(ratio, psnr, compress_time, per_sample_time_ms);
    }
  } catch (...) {
    // Return empty results on error
  }

  return results;
}

bool DenseNNPredictor::TrainRatioModel(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const TrainingConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Store architecture for save/load
  ratio_hidden_layers_ = config.hidden_layers;
  ratio_dropout_rate_ = config.dropout_rate;

  ratio_network_ = std::make_unique<MiniDNN::Network>();
  bool success = TrainNetwork(features, labels, config, *ratio_network_, ratio_scaler_);
  if (success) {
    ratio_model_ready_ = true;
  }
  return success;
}

bool DenseNNPredictor::TrainPSNRModel(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const TrainingConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Store architecture for save/load
  psnr_hidden_layers_ = config.hidden_layers;
  psnr_dropout_rate_ = config.dropout_rate;

  psnr_network_ = std::make_unique<MiniDNN::Network>();
  bool success = TrainNetwork(features, labels, config, *psnr_network_, psnr_scaler_);
  if (success) {
    psnr_model_ready_ = true;
  }
  return success;
}

bool DenseNNPredictor::TrainTimeModel(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const TrainingConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Store architecture for save/load
  time_hidden_layers_ = config.hidden_layers;
  time_dropout_rate_ = config.dropout_rate;

  time_network_ = std::make_unique<MiniDNN::Network>();
  bool success = TrainNetwork(features, labels, config, *time_network_, time_scaler_);
  if (success) {
    time_model_ready_ = true;
  }
  return success;
}

bool DenseNNPredictor::Train(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels) {
  return TrainWithConfig(features, labels, TrainingConfig());
}

bool DenseNNPredictor::TrainWithConfig(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels,
    const TrainingConfig& config) {
  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  // Extract individual label vectors
  std::vector<float> ratio_labels;
  std::vector<float> psnr_labels;
  std::vector<float> time_labels;
  ratio_labels.reserve(labels.size());
  psnr_labels.reserve(labels.size());
  time_labels.reserve(labels.size());

  for (const auto& label : labels) {
    ratio_labels.push_back(label.compression_ratio);
    psnr_labels.push_back(label.psnr_db);
    time_labels.push_back(label.compression_time_ms);
  }

  // Train all three models
  bool ratio_ok = TrainRatioModel(features, ratio_labels, config);
  if (!ratio_ok) {
    return false;
  }

  bool psnr_ok = TrainPSNRModel(features, psnr_labels, config);
  bool time_ok = TrainTimeModel(features, time_labels, config);

  // Return true if at least ratio model trained
  (void)psnr_ok;
  (void)time_ok;
  return ratio_ok;
}

Eigen::MatrixXd DenseNNPredictor::FeaturesToMatrix(
    const std::vector<CompressionFeatures>& features,
    const StandardScaler& scaler) {
  // Convert to vector of vectors
  std::vector<std::vector<float>> data;
  data.reserve(features.size());
  for (const auto& f : features) {
    data.push_back(f.ToVector());
  }

  // Normalize if scaler is fitted
  if (scaler.IsFitted()) {
    data = scaler.Transform(data);
  }

  // Convert to Eigen matrix (MiniDNN expects features x samples)
  size_t num_samples = data.size();
  size_t num_features = CompressionFeatures::NumFeatures();

  Eigen::MatrixXd matrix(num_features, num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    for (size_t j = 0; j < num_features; ++j) {
      matrix(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) =
          static_cast<double>(data[i][j]);
    }
  }

  return matrix;
}

void DenseNNPredictor::BuildNetwork(MiniDNN::Network& network,
                                     const std::vector<int>& hidden_layers,
                                     double /*dropout_rate*/) {
  int input_dim = static_cast<int>(CompressionFeatures::NumFeatures());
  int prev_dim = input_dim;

  // Add hidden layers with ReLU activation
  for (int units : hidden_layers) {
    network.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(prev_dim, units));
    prev_dim = units;
  }

  // Output layer (single neuron for regression) with Identity activation
  network.add_layer(new MiniDNN::FullyConnected<MiniDNN::Identity>(prev_dim, 1));

  // Set output layer for regression
  network.set_output(new MiniDNN::RegressionMSE());
}

bool DenseNNPredictor::TrainNetwork(
    const std::vector<CompressionFeatures>& features,
    const std::vector<float>& labels,
    const TrainingConfig& config,
    MiniDNN::Network& network,
    StandardScaler& scaler) {
  if (features.size() != labels.size() || features.empty()) {
    return false;
  }

  try {
    // Convert features to vectors
    std::vector<std::vector<float>> data;
    data.reserve(features.size());
    for (const auto& f : features) {
      data.push_back(f.ToVector());
    }

    // Fit scaler and transform
    auto normalized_data = scaler.FitTransform(data);

    // Create input matrix (features x samples)
    size_t num_samples = normalized_data.size();
    size_t num_features = CompressionFeatures::NumFeatures();

    Eigen::MatrixXd X(num_features, num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < num_features; ++j) {
        X(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) =
            static_cast<double>(normalized_data[i][j]);
      }
    }

    // Create label matrix (1 x samples)
    Eigen::MatrixXd y(1, num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      y(0, static_cast<Eigen::Index>(i)) = static_cast<double>(labels[i]);
    }

    // Build network architecture
    BuildNetwork(network, config.hidden_layers, config.dropout_rate);

    // Initialize parameters
    network.init(0, 0.01, 123);

    // Create optimizer (Adam)
    MiniDNN::Adam optimizer;
    optimizer.m_lrate = config.learning_rate;

    // Optional: set callback for verbose output
    MiniDNN::VerboseCallback callback;
    if (config.verbose) {
      network.set_callback(callback);
    }

    // Train the network
    network.fit(optimizer, X, y, config.batch_size, config.epochs, 123);

    return true;
  } catch (...) {
    return false;
  }
}

bool DenseNNPredictor::IsLossyCompression(const CompressionFeatures& features) const {
  return features.library_zfp_tol_0_01 > 0.5 ||
         features.library_zfp_tol_0_1 > 0.5;
}

// ============================================================================
// Reinforcement Learning Implementation
// ============================================================================

void DenseNNPredictor::RecordExperience(const RLExperience& experience) {
  std::lock_guard<std::mutex> lock(mutex_);

  experience_buffer_.push_back(experience);
  experience_count_++;

  // Limit buffer size (default 10000)
  while (experience_buffer_.size() > 10000) {
    experience_buffer_.pop_front();
  }
}

bool DenseNNPredictor::UpdateFromExperiences(const RLConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Need enough experiences for a batch
  if (experience_buffer_.size() < config.batch_size) {
    return false;
  }

  // Need a trained model to update
  if (!ratio_model_ready_ || !ratio_network_) {
    return false;
  }

  try {
    // Sample a random batch from experience buffer
    std::vector<CompressionFeatures> batch_features;
    std::vector<float> batch_ratio_labels;
    std::vector<float> batch_psnr_labels;
    std::vector<float> batch_time_labels;
    batch_features.reserve(config.batch_size);
    batch_ratio_labels.reserve(config.batch_size);
    batch_psnr_labels.reserve(config.batch_size);
    batch_time_labels.reserve(config.batch_size);

    std::uniform_int_distribution<size_t> dist(0, experience_buffer_.size() - 1);
    for (size_t i = 0; i < config.batch_size; ++i) {
      size_t idx = dist(rng_);
      const auto& exp = experience_buffer_[idx];
      batch_features.push_back(exp.features);
      batch_ratio_labels.push_back(static_cast<float>(exp.actual_ratio));
      batch_psnr_labels.push_back(static_cast<float>(exp.actual_psnr));
      batch_time_labels.push_back(static_cast<float>(exp.actual_compress_time));
    }

    // Convert features to matrix
    std::vector<std::vector<float>> data;
    data.reserve(batch_features.size());
    for (const auto& f : batch_features) {
      data.push_back(f.ToVector());
    }

    // Normalize using existing scaler
    auto normalized_data = ratio_scaler_.Transform(data);

    // Create input matrix (features x samples)
    size_t num_samples = normalized_data.size();
    size_t num_features = CompressionFeatures::NumFeatures();

    Eigen::MatrixXd X(num_features, num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < num_features; ++j) {
        X(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) =
            static_cast<double>(normalized_data[i][j]);
      }
    }

    // Create label matrix for ratio
    Eigen::MatrixXd y_ratio(1, num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      y_ratio(0, static_cast<Eigen::Index>(i)) =
          static_cast<double>(batch_ratio_labels[i]);
    }

    // Perform a few gradient updates (online learning)
    MiniDNN::Adam optimizer;
    optimizer.m_lrate = config.learning_rate;

    // Do a single epoch with the batch (incremental update)
    ratio_network_->fit(optimizer, X, y_ratio,
                        static_cast<int>(config.batch_size), 1, -1);

    // Update PSNR model if ready
    if (psnr_model_ready_ && psnr_network_) {
      auto psnr_normalized = psnr_scaler_.Transform(data);
      Eigen::MatrixXd X_psnr(num_features, num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
          X_psnr(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) =
              static_cast<double>(psnr_normalized[i][j]);
        }
      }

      Eigen::MatrixXd y_psnr(1, num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        y_psnr(0, static_cast<Eigen::Index>(i)) =
            static_cast<double>(batch_psnr_labels[i]);
      }

      psnr_network_->fit(optimizer, X_psnr, y_psnr,
                         static_cast<int>(config.batch_size), 1, -1);
    }

    // Update time model if ready
    if (time_model_ready_ && time_network_) {
      auto time_normalized = time_scaler_.Transform(data);
      Eigen::MatrixXd X_time(num_features, num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
          X_time(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) =
              static_cast<double>(time_normalized[i][j]);
        }
      }

      Eigen::MatrixXd y_time(1, num_samples);
      for (size_t i = 0; i < num_samples; ++i) {
        y_time(0, static_cast<Eigen::Index>(i)) =
            static_cast<double>(batch_time_labels[i]);
      }

      time_network_->fit(optimizer, X_time, y_time,
                         static_cast<int>(config.batch_size), 1, -1);
    }

    // Decay epsilon
    epsilon_ = std::max(config.min_epsilon, epsilon_ * config.epsilon_decay);

    return true;
  } catch (...) {
    return false;
  }
}

void DenseNNPredictor::SetExplorationMode(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  exploration_enabled_ = enable;
}

double DenseNNPredictor::GetExplorationRate() const {
  return epsilon_;
}

size_t DenseNNPredictor::GetExperienceCount() const {
  return experience_count_;
}

void DenseNNPredictor::ClearExperiences() {
  std::lock_guard<std::mutex> lock(mutex_);
  experience_buffer_.clear();
  experience_count_ = 0;
}

}  // namespace compress
}  // namespace hshm
