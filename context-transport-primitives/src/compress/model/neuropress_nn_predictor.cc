/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file neuropress_nn_predictor.cc
 * @brief NeuroPress dense NN predictor implementation (CPU inference).
 */

#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace ctp::compress::model {

NeuroPressNNPredictor::NeuroPressNNPredictor() = default;

bool NeuroPressNNPredictor::Load(const std::string& model_dir) {
  std::string path = model_dir;
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  path += "model.nnwt";

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }

  // Read header (24 bytes): magic, version, num_layers, input_dim, hidden_dim,
  // output_dim
  uint32_t magic, version, num_layers, input_dim, hidden_dim, output_dim;
  file.read(reinterpret_cast<char*>(&magic), 4);
  file.read(reinterpret_cast<char*>(&version), 4);
  file.read(reinterpret_cast<char*>(&num_layers), 4);
  file.read(reinterpret_cast<char*>(&input_dim), 4);
  file.read(reinterpret_cast<char*>(&hidden_dim), 4);
  file.read(reinterpret_cast<char*>(&output_dim), 4);

  if (!file || magic != 0x4E4E5754 || version != 2) {
    return false;
  }

  // Validate architecture
  if (input_dim != kInputDim || hidden_dim != kHiddenDim ||
      output_dim != kOutputDim || num_layers != kNumLayers) {
    return false;
  }

  // Read normalization (8+8+8+8 = 32 floats)
  x_means_.resize(input_dim);
  x_stds_.resize(input_dim);
  y_means_.resize(output_dim);
  y_stds_.resize(output_dim);

  file.read(reinterpret_cast<char*>(x_means_.data()), input_dim * 4);
  file.read(reinterpret_cast<char*>(x_stds_.data()), input_dim * 4);
  file.read(reinterpret_cast<char*>(y_means_.data()), output_dim * 4);
  file.read(reinterpret_cast<char*>(y_stds_.data()), output_dim * 4);

  if (!file) {
    return false;
  }

  // Layer dimensions: [8,64], [64,64], [64,64], [64,64], [64,8]
  std::vector<uint32_t> fan_ins = {input_dim, hidden_dim, hidden_dim,
                                    hidden_dim, hidden_dim};
  std::vector<uint32_t> fan_outs = {hidden_dim, hidden_dim, hidden_dim,
                                     hidden_dim, output_dim};

  // Calculate total size needed
  size_t total_weights = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    total_weights += static_cast<size_t>(fan_ins[i]) * fan_outs[i];
  }
  size_t total_biases = hidden_dim * 4 + output_dim;  // 64*4 + 8

  weights_.clear();
  biases_.clear();
  weight_offsets_.clear();
  bias_offsets_.clear();

  weights_.reserve(total_weights);
  biases_.reserve(total_biases);
  weight_offsets_.reserve(num_layers);
  bias_offsets_.reserve(num_layers);

  // Read weights and biases layer by layer
  for (uint32_t i = 0; i < num_layers; ++i) {
    uint32_t w_count = fan_ins[i] * fan_outs[i];
    uint32_t b_count = fan_outs[i];

    weight_offsets_.push_back(weights_.size());
    bias_offsets_.push_back(biases_.size());

    // Read weight matrix
    std::vector<float> w(w_count);
    file.read(reinterpret_cast<char*>(w.data()), w_count * 4);
    weights_.insert(weights_.end(), w.begin(), w.end());

    // Read bias vector
    std::vector<float> b(b_count);
    file.read(reinterpret_cast<char*>(b.data()), b_count * 4);
    biases_.insert(biases_.end(), b.begin(), b.end());

    if (!file) {
      return false;
    }
  }

  // Read feature bounds (v2+): x_mins[8], x_maxs[8]
  x_mins_.resize(input_dim);
  x_maxs_.resize(input_dim);
  file.read(reinterpret_cast<char*>(x_mins_.data()), input_dim * 4);
  file.read(reinterpret_cast<char*>(x_maxs_.data()), input_dim * 4);

  if (!file) {
    return false;
  }

  is_ready_ = true;
  return true;
}

bool NeuroPressNNPredictor::Save(const std::string& model_dir) {
  if (!is_ready_ || weights_.empty() || biases_.empty()) {
    return false;
  }

  std::string path = model_dir;
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  path += "model.nnwt";

  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }

  // Write header
  uint32_t magic = 0x4E4E5754;
  uint32_t version = 2;
  uint32_t num_layers = kNumLayers;
  uint32_t input_dim = kInputDim;
  uint32_t hidden_dim = kHiddenDim;
  uint32_t output_dim = kOutputDim;

  file.write(reinterpret_cast<const char*>(&magic), 4);
  file.write(reinterpret_cast<const char*>(&version), 4);
  file.write(reinterpret_cast<const char*>(&num_layers), 4);
  file.write(reinterpret_cast<const char*>(&input_dim), 4);
  file.write(reinterpret_cast<const char*>(&hidden_dim), 4);
  file.write(reinterpret_cast<const char*>(&output_dim), 4);

  // Write normalization
  file.write(reinterpret_cast<const char*>(x_means_.data()), input_dim * 4);
  file.write(reinterpret_cast<const char*>(x_stds_.data()), input_dim * 4);
  file.write(reinterpret_cast<const char*>(y_means_.data()), output_dim * 4);
  file.write(reinterpret_cast<const char*>(y_stds_.data()), output_dim * 4);

  // Write weights and biases
  std::vector<uint32_t> fan_ins = {input_dim, hidden_dim, hidden_dim,
                                    hidden_dim, hidden_dim};
  std::vector<uint32_t> fan_outs = {hidden_dim, hidden_dim, hidden_dim,
                                     hidden_dim, output_dim};
  for (size_t i = 0; i < num_layers; ++i) {
    size_t w_offset = weight_offsets_[i];
    size_t w_count = static_cast<size_t>(fan_ins[i]) * fan_outs[i];
    file.write(reinterpret_cast<const char*>(&weights_[w_offset]), w_count * 4);

    size_t b_offset = bias_offsets_[i];
    size_t b_count = fan_outs[i];
    file.write(reinterpret_cast<const char*>(&biases_[b_offset]), b_count * 4);
  }

  // Write feature bounds
  file.write(reinterpret_cast<const char*>(x_mins_.data()), input_dim * 4);
  file.write(reinterpret_cast<const char*>(x_maxs_.data()), input_dim * 4);

  return static_cast<bool>(file);
}

bool NeuroPressNNPredictor::IsReady() const { return is_ready_; }

std::vector<float> NeuroPressNNPredictor::FeaturesTo8Input(
    const CompressionFeatures& features) const {
  // NeuroPress expects: [algo_id, quant, shuffle, error_bound, data_size,
  // entropy, mad, second_derivative]
  // Map library_config_id to algo_id (0-7): library_id / 10
  float algo_id = static_cast<float>(static_cast<int>(features.library_config_id) / 10 % 8);
  if (algo_id < 0) algo_id = 0;
  if (algo_id >= 8) algo_id = 7;

  return {algo_id,
          0.0f,  // quant (0 for our interface)
          0.0f,  // shuffle (0 for our interface)
          0.0f,  // error_bound (0 for lossless)
          static_cast<float>(features.chunk_size_bytes),
          static_cast<float>(features.shannon_entropy),
          static_cast<float>(features.mad),
          static_cast<float>(features.second_derivative_mean)};
}

std::vector<float> NeuroPressNNPredictor::Standardize(
    const std::vector<float>& x) const {
  std::vector<float> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    float std_val = std::max(x_stds_[i], 1e-8f);
    result[i] = (x[i] - x_means_[i]) / std_val;
  }
  return result;
}

std::vector<float> NeuroPressNNPredictor::ForwardPass(
    const std::vector<float>& x_norm) const {
  std::vector<uint32_t> fan_ins = {kInputDim, kHiddenDim, kHiddenDim,
                                    kHiddenDim, kHiddenDim};
  std::vector<uint32_t> fan_outs = {kHiddenDim, kHiddenDim, kHiddenDim,
                                     kHiddenDim, kOutputDim};

  std::vector<float> x = x_norm;

  // Forward through 5 layers
  for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
    uint32_t fan_in = fan_ins[layer];
    uint32_t fan_out = fan_outs[layer];

    size_t w_offset = weight_offsets_[layer];
    size_t b_offset = bias_offsets_[layer];

    std::vector<float> y(fan_out, 0.0f);

    // Matrix-vector multiplication: y = W @ x + b
    for (uint32_t i = 0; i < fan_out; ++i) {
      float sum = biases_[b_offset + i];
      for (uint32_t j = 0; j < fan_in; ++j) {
        sum += weights_[w_offset + i * fan_in + j] * x[j];
      }
      y[i] = sum;
    }

    // ReLU on hidden layers, identity on output
    if (layer < kNumLayers - 1) {
      for (uint32_t i = 0; i < fan_out; ++i) {
        y[i] = std::max(0.0f, y[i]);
      }
    }

    x = y;
  }

  return x;
}

std::vector<float> NeuroPressNNPredictor::InverseTransform(
    const std::vector<float>& y_norm) const {
  std::vector<float> result(y_norm.size());
  for (size_t i = 0; i < y_norm.size(); ++i) {
    result[i] = y_norm[i] * y_stds_[i] + y_means_[i];
  }
  return result;
}

CompressionPrediction NeuroPressNNPredictor::Predict(
    const CompressionFeatures& features) {
  if (!is_ready_) {
    return CompressionPrediction();
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Map features to 8-input vector
  auto x = FeaturesTo8Input(features);

  // Standardize
  auto x_norm = Standardize(x);

  // Forward pass
  auto y_norm = ForwardPass(x_norm);

  // Inverse transform
  auto y = InverseTransform(y_norm);

  auto end_time = std::chrono::high_resolution_clock::now();
  double infer_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();

  // NeuroPress outputs: [comp_time, decomp_time, ratio, psnr, ...]
  // Map to CompressionPrediction
  double comp_time = y[0];
  double ratio = y[2];
  double psnr = y[3];

  // Clamp: comp_time >= 1ms, ratio in (0, 100]
  comp_time = std::max(1.0, comp_time);
  ratio = std::max(0.001, std::min(100.0, ratio));

  return CompressionPrediction(ratio, psnr, comp_time, infer_ms);
}

}  // namespace ctp::compress::model
