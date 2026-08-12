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
#include <array>
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

  // Online-learning state: fresh on every Load(), matching upstream's
  // per-process device-resident state (a reload starts learning over).
  // log_var_ = 0 -> initial precision weight exp(-0.5*0) = 1.0 (neutral).
  log_var_.assign(output_dim, 0.0f);
  ema_weights_.assign(weights_.size(), 0.0f);
  ema_biases_.assign(biases_.size(), 0.0f);
  sgd_call_count_ = 0;

#if CTP_ENABLE_NEUROPRESS_GPU
  // Upload to device -- this is what actually makes Predict()/Train() run
  // as CUDA kernels below, matching NeuroPress's own device-resident
  // design. gpu_weights_ stays null (falls back to the CPU path this class
  // already has) if CUDA is compiled in but no device is available at
  // runtime -- e.g. a CUDA-built binary running on a CPU-only host.
  gpu::NeuroPressGpuWeights* raw = gpu::NeuroPressGpuLoad(
      weights_.data(), weights_.size(), biases_.data(), biases_.size(),
      x_means_.data(), x_stds_.data(), y_means_.data(), y_stds_.data());
  gpu_weights_.reset(raw, [](gpu::NeuroPressGpuWeights* p) {
    gpu::NeuroPressGpuFree(p);
  });
#endif

  is_ready_ = true;
  return true;
}

bool NeuroPressNNPredictor::Save(const std::string& model_dir) {
  if (!is_ready_ || weights_.empty() || biases_.empty()) {
    return false;
  }

#if CTP_ENABLE_NEUROPRESS_GPU
  // weights_/biases_ are a stale snapshot from Load() time once training
  // has run on the GPU -- pull the current (trained) values back before
  // writing the file, or Save() would silently persist the untrained model.
  if (gpu_weights_) {
    gpu::NeuroPressGpuDownloadWeights(gpu_weights_.get(), weights_.data(),
                                      biases_.data());
  }
#endif

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

namespace {

// Maps a CompressionFactory base_id to NeuroPress's trained one-hot algo_id
// (0-7), matching neural_net/core/configs.py's ALGORITHM_NAMES order
// exactly: ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans',
// 'cascaded', 'bitcomp']. NeuroPress's action space is ONLY these 8 nvcomp
// algorithms -- base_id here must be the exact nvcomp base_id from
// ranking.h's KnownCompressors(), not derived arithmetically (the previous
// `base_id % 8` collided nvcomp-zstd(15) with nvcomp-cascaded(23), and
// nvcomp-gdeflate(16) with nvcomp-bitcomp(24), since both pairs share a
// remainder mod 8 despite being distinct trained indices).
int NeuroPressAlgoIdForBaseId(int base_id) {
  switch (base_id) {
    case 13: return 0;  // nvcomp-lz4
    case 14: return 1;  // nvcomp-snappy
    case 17: return 2;  // nvcomp-deflate
    case 16: return 3;  // nvcomp-gdeflate
    case 15: return 4;  // nvcomp-zstd
    case 18: return 5;  // nvcomp-ans
    case 23: return 6;  // nvcomp-cascaded
    case 24: return 7;  // nvcomp-bitcomp
    default:
      // Outside NeuroPress's trained action space entirely (CPU libraries,
      // cusz/ndzip/cuszp/zfp-sycl): the model has no representation for
      // these algorithms at all, so there is no "correct" index -- fall
      // back to a deterministic, in-range value rather than crashing or
      // silently colliding with one of the 8 real trained algorithms above.
      return ((base_id % 8) + 8) % 8;
  }
}

}  // namespace

std::vector<float> NeuroPressNNPredictor::FeaturesTo8Input(
    const CompressionFeatures& features) const {
  // NeuroPress expects: [algo_id, quant, shuffle, error_bound, data_size,
  // entropy, mad, second_derivative]
  int base_id = static_cast<int>(features.library_config_id) / 10;
  float algo_id = static_cast<float>(NeuroPressAlgoIdForBaseId(base_id));

  return {algo_id,
          static_cast<float>(features.quantize),      // quant preprocessor
          static_cast<float>(features.byte_shuffle),  // shuffle preprocessor
          static_cast<float>(features.error_bound),   // lossy/quant bound
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

NeuroPressNNPredictor::SGDActivations NeuroPressNNPredictor::ForwardWithCache(
    const std::vector<float>& x_norm) const {
  SGDActivations a;
  a.x = x_norm;

  size_t w0 = weight_offsets_[0], b0 = bias_offsets_[0];
  a.z1.resize(kHiddenDim);
  a.h1.resize(kHiddenDim);
  for (uint32_t i = 0; i < kHiddenDim; ++i) {
    float sum = biases_[b0 + i];
    for (uint32_t j = 0; j < kInputDim; ++j) {
      sum += weights_[w0 + i * kInputDim + j] * a.x[j];
    }
    a.z1[i] = sum;
    a.h1[i] = std::max(0.0f, sum);
  }

  auto hidden_layer = [&](const std::vector<float>& h_in, int layer_idx,
                          std::vector<float>& z_out, std::vector<float>& h_out) {
    size_t w = weight_offsets_[layer_idx], b = bias_offsets_[layer_idx];
    z_out.resize(kHiddenDim);
    h_out.resize(kHiddenDim);
    for (uint32_t i = 0; i < kHiddenDim; ++i) {
      float sum = biases_[b + i];
      for (uint32_t j = 0; j < kHiddenDim; ++j) {
        sum += weights_[w + i * kHiddenDim + j] * h_in[j];
      }
      z_out[i] = sum;
      h_out[i] = std::max(0.0f, sum);
    }
  };
  hidden_layer(a.h1, 1, a.z2, a.h2);
  hidden_layer(a.h2, 2, a.z3, a.h3);
  hidden_layer(a.h3, 3, a.z4, a.h4);

  size_t w4 = weight_offsets_[4], b4 = bias_offsets_[4];
  a.y.resize(kOutputDim);
  for (uint32_t i = 0; i < kOutputDim; ++i) {
    float sum = biases_[b4 + i];
    for (uint32_t j = 0; j < kHiddenDim; ++j) {
      sum += weights_[w4 + i * kHiddenDim + j] * a.h4[j];
    }
    a.y[i] = sum;
  }
  return a;
}

std::vector<float> NeuroPressNNPredictor::InverseTransform(
    const std::vector<float>& y_norm) const {
  std::vector<float> result(y_norm.size());
  for (size_t i = 0; i < y_norm.size(); ++i) {
    result[i] = y_norm[i] * y_stds_[i] + y_means_[i];
  }
  // Outputs 0,1,2 (comp_time, decomp_time, ratio) and 4,5,6 (rmse, max_error,
  // mae) are trained in log1p() space (neural_net/core/data.py's
  // OUTPUT_INVERSE); NeuroPress's own CUDA inference undoes this with
  // expm1f (src/nn/nn_gpu.cu). Without it, every consumer of these fields
  // (compression ratio, compression/decompression time) receives the
  // log-space value instead of the real one -- de-correlated from actual
  // units, and wrong whenever compared or combined with a linear-space
  // quantity (e.g. EstWorkflowCompressTime() mixing predicted ratio with
  // real chunk_size/bandwidth). Output 3 (psnr) is linear/identity; output
  // 7 (ssim_nlog) needs a different, non-expm1 inverse this predictor
  // doesn't currently expose (Predict() never reads index 7).
  for (int i : {0, 1, 2, 4, 5, 6}) {
    if (static_cast<size_t>(i) < result.size()) {
      result[i] = std::expm1(result[i]);
    }
  }
  return result;
}

CompressionPrediction NeuroPressNNPredictor::Predict(
    const CompressionFeatures& features) {
  if (!is_ready_) {
    return CompressionPrediction();
  }
  auto batch = PredictBatch({features});
  return batch.empty() ? CompressionPrediction() : batch.front();
}

std::vector<CompressionPrediction> NeuroPressNNPredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  if (batch.empty()) {
    return {};
  }
  if (!is_ready_) {
    // Base class contract (see predictor.h's default PredictBatch(), and
    // Rank(), which indexes preds[i] against candidates[i] unconditionally):
    // callers assume the returned vector is always exactly batch.size()
    // long, even for a not-ready model -- Predict() itself already returns
    // a default CompressionPrediction() in that case, so match it here.
    return std::vector<CompressionPrediction>(batch.size());
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());

#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    // Real batched path: one kernel launch scores the WHOLE candidate set
    // (mirrors NeuroPress's nnFusedInferenceKernel) -- the host only
    // touches the small raw 8-dim inputs and the 4-value outputs, never
    // the weights or activations.
    std::vector<float> raw_inputs(batch.size() * kInputDim);
    for (size_t i = 0; i < batch.size(); ++i) {
      auto x = FeaturesTo8Input(batch[i]);
      std::copy(x.begin(), x.end(), raw_inputs.begin() +
                static_cast<long>(i * kInputDim));
    }
    std::vector<float> comp_time(batch.size()), decomp_time(batch.size()),
        ratio(batch.size()), psnr(batch.size());
    if (!gpu::NeuroPressGpuInferBatch(gpu_weights_.get(), raw_inputs.data(),
                                      static_cast<int>(batch.size()),
                                      comp_time.data(), decomp_time.data(),
                                      ratio.data(), psnr.data())) {
      // Device inference failed. Returning the zero-filled buffers would
      // hand the caller a full, well-formed ranking built from nothing, so
      // return empty instead and let it fall back -- the same contract
      // EstCompressionStats already honors for a stats failure.
      return {};
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(
                          end_time - start_time)
                          .count() /
                      static_cast<double>(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      // Already clamped inside the kernel (nn_gpu.cu:217-231 order: sanity
      // ceiling first, then the policy floors/caps). Repeat the policy half
      // here so the CPU and GPU paths read identically.
      results.emplace_back(
          std::max(0.1, std::min(100.0, static_cast<double>(ratio[i]))),
          std::max(0.0, std::min(120.0, static_cast<double>(psnr[i]))),
          std::max(1.0, static_cast<double>(comp_time[i])),
          std::max(1.0, static_cast<double>(decomp_time[i])), infer_ms);
    }
    return results;
  }
#endif

  // CPU fallback: same per-candidate math as before, looped.
  for (const auto& features : batch) {
    auto x = FeaturesTo8Input(features);
    auto x_norm = Standardize(x);
    auto y_norm = ForwardPass(x_norm);
    auto y = InverseTransform(y_norm);

    // Same policy clamps as the GPU path above (nn_gpu.cu). y[1] is the
    // NN's own DEcompression-time output, distinct from y[0]; it is what
    // the cost model's w1 term ranks on, so it must not be aliased to the
    // compression time.
    // Sanity ceiling first, then the policy floors/caps -- nn_gpu.cu:217-231.
    // The 1e6 ceiling is load-bearing under NaN: std::min(NaN, 1e6) returns
    // NaN here (unlike CUDA's fminf), but the subsequent std::max(1.0, NaN)
    // would otherwise make a drifted head the CHEAPEST candidate rather than
    // the most expensive, so guard the non-finite case explicitly.
    auto sane = [](double v, double lo, double hi, double fallback) {
      if (!std::isfinite(v)) return fallback;
      return std::max(lo, std::min(v, hi));
    };
    double comp_time = std::max(1.0, sane(y[0], 1e-6, 1e6, 1e6));
    double decomp_time = std::max(1.0, sane(y[1], 1e-6, 1e6, 1e6));
    double ratio = std::min(100.0, sane(y[2], 0.1, 1e5, 0.1));
    double psnr = sane(y[3], 0.0, 120.0, 0.0);
    results.emplace_back(ratio, psnr, comp_time, decomp_time, 0.0);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  double infer_ms = std::chrono::duration<double, std::milli>(
                        end_time - start_time)
                        .count() /
                    static_cast<double>(batch.size());
  for (auto& r : results) r.inference_time_ms = infer_ms;
  return results;
}

bool NeuroPressNNPredictor::Train(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels) {
  if (!is_ready_ || features.empty() || features.size() != labels.size()) {
    return false;
  }
  const size_t num_samples =
      std::min(features.size(), static_cast<size_t>(kMaxSGDSamples));

#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    // Real SGD kernel: forward + backward + weight update entirely
    // on-device (see neuropress_nn_gpu_kernels.cu's SGDKernel), mirroring
    // nnSGDKernel exactly rather than the CPU port below.
    std::vector<gpu::NeuroPressGpuSGDSample> gpu_samples(num_samples);
    for (size_t si = 0; si < num_samples; ++si) {
      auto x = FeaturesTo8Input(features[si]);
      std::copy(x.begin(), x.end(), gpu_samples[si].raw_input);
      gpu_samples[si].actual_ratio = labels[si].compression_ratio;
      gpu_samples[si].actual_comp_time_ms = labels[si].compression_time_ms;
      gpu_samples[si].actual_decomp_time_ms = labels[si].decompression_time_ms;
      gpu_samples[si].actual_psnr_db = labels[si].psnr_db;
    }
    return gpu::NeuroPressGpuTrain(gpu_weights_.get(), gpu_samples.data(),
                                   static_cast<int>(num_samples));
  }
#endif

  // ---- CPU fallback (used when CUDA is unavailable or no device was
  // present at Load() time) ----
  // Per-sample forward pass + target/error computation (nn_gpu.cu's
  // per-sample loop, lines ~690-853) ----
  std::vector<SGDActivations> acts(num_samples);
  std::vector<std::array<float, kOutputDim>> d5_clamped(num_samples);
  std::vector<std::array<float, kOutputDim>> d5_raw(num_samples);

  for (size_t si = 0; si < num_samples; ++si) {
    auto x = FeaturesTo8Input(features[si]);
    auto x_norm = Standardize(x);
    acts[si] = ForwardWithCache(x_norm);
    const auto& y = acts[si].y;
    const TrainingLabels& lbl = labels[si];

    std::array<float, kOutputDim> d5{};

    // Output 2: compression ratio (log1p target, always trained).
    {
      float clamped = std::max(0.5f, std::min(lbl.compression_ratio, 10000.0f));
      float y_std2 = std::max(y_stds_[2], 1e-8f);
      d5[2] = y[2] - (std::log1p(clamped) - y_means_[2]) / y_std2;
    }
    // Output 0: compression time (0 = not measured, skip).
    if (lbl.compression_time_ms > 0.0f) {
      float clamped = std::max(0.01f, std::min(lbl.compression_time_ms, 5000.0f));
      float y_std0 = std::max(y_stds_[0], 1e-8f);
      d5[0] = y[0] - (std::log1p(clamped) - y_means_[0]) / y_std0;
    } else {
      d5[0] = 0.0f;
    }
    // Output 1: decompression time (0 = not measured, skip).
    if (lbl.decompression_time_ms > 0.0f) {
      float clamped = std::max(0.01f, std::min(lbl.decompression_time_ms, 5000.0f));
      float y_std1 = std::max(y_stds_[1], 1e-8f);
      d5[1] = y[1] - (std::log1p(clamped) - y_means_[1]) / y_std1;
    } else {
      d5[1] = 0.0f;
    }
    // Output 3: PSNR (negative = not applicable, skip; 0 = lossless -> 120dB).
    if (lbl.psnr_db >= 0.0f) {
      float psnr_val = (lbl.psnr_db == 0.0f) ? 120.0f : lbl.psnr_db;
      float clamped_psnr = std::min(psnr_val, 120.0f);
      float y_std3 = std::max(y_stds_[3], 1e-8f);
      d5[3] = y[3] - (clamped_psnr - y_means_[3]) / y_std3;
    } else {
      d5[3] = 0.0f;
    }
    // Outputs 4-7 (rmse/max_error/mae/ssim) are not trained by online SGD,
    // matching upstream -- those are offline-trainer-only labels.
    for (int o = 4; o < static_cast<int>(kOutputDim); ++o) d5[o] = 0.0f;

    // Noise gate: a comp_time error under ~10% of a std-dev is GPU/host
    // timing jitter, not a real prediction error -- don't train on it.
    constexpr float kNoiseGateThresh = 0.10f;
    if (std::fabs(d5[0]) < kNoiseGateThresh) d5[0] = 0.0f;

    d5_raw[si] = d5;  // RAW error, cached for uncertainty weighting below

    // Clamp to +-0.5 std-dev (Huber-style) so a single OOD sample can't
    // cause a catastrophic weight jump.
    constexpr float kSgdErrorDelta = 0.5f;
    for (int o = 0; o < static_cast<int>(kOutputDim); ++o) {
      d5[o] = std::max(-kSgdErrorDelta, std::min(d5[o], kSgdErrorDelta));
    }
    d5_clamped[si] = d5;
  }

  // ---- Phase 1.5: Uncertainty weighting (Kendall et al., 2018) ----
  // Each output has a learned log_var[o]; precision = exp(-log_var[o]).
  // Noisy outputs (large raw MSE) get down-weighted before backprop so they
  // don't destabilize the shared hidden layers.
  constexpr float kUwLr = 0.01f;
  constexpr float kUwLogVarMin = -2.0f;
  constexpr float kUwLogVarMax = 4.0f;
  for (int o = 0; o < static_cast<int>(kOutputDim); ++o) {
    float lv = log_var_[o];
    float precision = std::exp(std::max(-20.0f, std::min(20.0f, -lv)));
    float raw_mse = 0.0f;
    for (size_t si = 0; si < num_samples; ++si) {
      float e = d5_raw[si][o];
      raw_mse += e * e;
    }
    raw_mse /= static_cast<float>(num_samples);
    float grad_lv = 0.5f * (1.0f - precision * raw_mse);
    lv -= kUwLr * grad_lv;
    lv = std::max(kUwLogVarMin, std::min(lv, kUwLogVarMax));
    log_var_[o] = lv;
    float uw = std::exp(std::max(-20.0f, std::min(20.0f, -0.5f * lv)));
    for (size_t si = 0; si < num_samples; ++si) {
      d5_clamped[si][o] *= uw;
    }
  }

  // ---- Phase 2: per-output backward passes with PCGrad-lite, accumulated
  // into a combined (region-0-equivalent) gradient ----
  constexpr float kGradClipThreshold = 0.1f;
  constexpr float kPcgradCosThresh = -0.1f;
  constexpr float kDefaultLearningRate = 0.01f;  // NeuroPress's g_reinforce_lr default

  std::vector<float> combined_dw(weights_.size(), 0.0f);
  std::vector<float> combined_db(biases_.size(), 0.0f);

  for (int target_out = 0; target_out < static_cast<int>(kOutputDim); ++target_out) {
    std::vector<float> out_dw(weights_.size(), 0.0f);
    std::vector<float> out_db(biases_.size(), 0.0f);

    for (size_t si = 0; si < num_samples; ++si) {
      const auto& a = acts[si];

      // Step 1: L4 backward delta for ALL outputs (needed for PCGrad),
      // normalized to unit vectors for scale-invariant cosine comparison.
      std::array<std::array<float, kHiddenDim>, kOutputDim> dz4_all{};
      for (int o = 0; o < static_cast<int>(kOutputDim); ++o) {
        float es = d5_clamped[si][o];
        float norm_sq = 0.0f;
        for (uint32_t t = 0; t < kHiddenDim; ++t) {
          float dh4_t = weights_[weight_offsets_[4] + o * kHiddenDim + t] * es;
          float v = (a.z4[t] > 0.0f) ? dh4_t : 0.0f;
          dz4_all[o][t] = v;
          norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq) + 1e-6f;
        for (uint32_t t = 0; t < kHiddenDim; ++t) dz4_all[o][t] /= norm;
      }

      // Step 2: PCGrad projection -- only project against outputs whose
      // gradient direction truly conflicts (cosine < -0.1), ignoring noise
      // within that margin (matters for single/few-sample SGD).
      std::array<float, kHiddenDim> my_dz4;
      for (uint32_t t = 0; t < kHiddenDim; ++t) my_dz4[t] = dz4_all[target_out][t];
      for (int j = 0; j < static_cast<int>(kOutputDim); ++j) {
        if (j == target_out) continue;
        float cos_ij = 0.0f;
        for (uint32_t t = 0; t < kHiddenDim; ++t) cos_ij += my_dz4[t] * dz4_all[j][t];
        if (cos_ij < kPcgradCosThresh) {
          for (uint32_t t = 0; t < kHiddenDim; ++t) my_dz4[t] -= cos_ij * dz4_all[j][t];
        }
      }
      float err_mag = std::fabs(d5_clamped[si][target_out]);
      std::array<float, kHiddenDim> dz4;
      for (uint32_t t = 0; t < kHiddenDim; ++t) dz4[t] = my_dz4[t] * err_mag;

      // Step 3: W5/b5 gradient uses the ORIGINAL (unprojected) error.
      float error_signal = d5_clamped[si][target_out];
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        out_dw[weight_offsets_[4] + target_out * kHiddenDim + t] +=
            error_signal * a.h4[t];
      }
      out_db[bias_offsets_[4] + target_out] += error_signal;

      // Step 4/4b/4c: L4 gradient + backward L4->L3 (using PROJECTED dz4).
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        for (uint32_t i = 0; i < kHiddenDim; ++i)
          out_dw[weight_offsets_[3] + t * kHiddenDim + i] += dz4[t] * a.h3[i];
        out_db[bias_offsets_[3] + t] += dz4[t];
      }
      std::array<float, kHiddenDim> dz3{};
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        float dh3_t = 0.0f;
        for (uint32_t j = 0; j < kHiddenDim; ++j)
          dh3_t += weights_[weight_offsets_[3] + j * kHiddenDim + t] * dz4[j];
        dz3[t] = (a.z3[t] > 0.0f) ? dh3_t : 0.0f;
      }

      // Step 5/5b: L3 gradient + backward L3->L2.
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        for (uint32_t i = 0; i < kHiddenDim; ++i)
          out_dw[weight_offsets_[2] + t * kHiddenDim + i] += dz3[t] * a.h2[i];
        out_db[bias_offsets_[2] + t] += dz3[t];
      }
      std::array<float, kHiddenDim> dz2{};
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        float dh2_t = 0.0f;
        for (uint32_t j = 0; j < kHiddenDim; ++j)
          dh2_t += weights_[weight_offsets_[2] + j * kHiddenDim + t] * dz3[j];
        dz2[t] = (a.z2[t] > 0.0f) ? dh2_t : 0.0f;
      }

      // Step 6/6b: L2 gradient + backward L2->L1, then L1 gradient.
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        for (uint32_t i = 0; i < kHiddenDim; ++i)
          out_dw[weight_offsets_[1] + t * kHiddenDim + i] += dz2[t] * a.h1[i];
        out_db[bias_offsets_[1] + t] += dz2[t];
      }
      std::array<float, kHiddenDim> dz1{};
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        float dh1_t = 0.0f;
        for (uint32_t j = 0; j < kHiddenDim; ++j)
          dh1_t += weights_[weight_offsets_[1] + j * kHiddenDim + t] * dz2[j];
        dz1[t] = (a.z1[t] > 0.0f) ? dh1_t : 0.0f;
      }
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        for (uint32_t i = 0; i < kInputDim; ++i)
          out_dw[weight_offsets_[0] + t * kInputDim + i] += dz1[t] * a.x[i];
        out_db[bias_offsets_[0] + t] += dz1[t];
      }
    }  // per-sample

    // Average this output's gradient over samples, clip by its own norm
    // (only the parameters it actually touches: shared L1-L4 + its own
    // W5/b5 row), then fold the (learning-rate + clip-scaled) result into
    // the combined gradient.
    float inv_n = 1.0f / static_cast<float>(num_samples);
    double norm_sq = 0.0;
    for (uint32_t layer = 0; layer < 4; ++layer) {
      size_t wsz = (layer == 0) ? size_t{kHiddenDim} * kInputDim
                                 : size_t{kHiddenDim} * kHiddenDim;
      size_t wo = weight_offsets_[layer], bo = bias_offsets_[layer];
      for (size_t k = 0; k < wsz; ++k) {
        out_dw[wo + k] *= inv_n;
        norm_sq += static_cast<double>(out_dw[wo + k]) * out_dw[wo + k];
      }
      for (uint32_t k = 0; k < kHiddenDim; ++k) {
        out_db[bo + k] *= inv_n;
        norm_sq += static_cast<double>(out_db[bo + k]) * out_db[bo + k];
      }
    }
    {
      size_t wo = weight_offsets_[4], bo = bias_offsets_[4];
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        size_t idx = wo + target_out * kHiddenDim + t;
        out_dw[idx] *= inv_n;
        norm_sq += static_cast<double>(out_dw[idx]) * out_dw[idx];
      }
      out_db[bo + target_out] *= inv_n;
      norm_sq += static_cast<double>(out_db[bo + target_out]) *
                out_db[bo + target_out];
    }
    float out_norm = static_cast<float>(std::sqrt(norm_sq)) + 1e-8f;
    float clip_scale =
        (out_norm > kGradClipThreshold) ? (kGradClipThreshold / out_norm) : 1.0f;
    float lr_out = kDefaultLearningRate * clip_scale;

    for (uint32_t layer = 0; layer < 4; ++layer) {
      size_t wsz = (layer == 0) ? size_t{kHiddenDim} * kInputDim
                                 : size_t{kHiddenDim} * kHiddenDim;
      size_t wo = weight_offsets_[layer], bo = bias_offsets_[layer];
      for (size_t k = 0; k < wsz; ++k) combined_dw[wo + k] += lr_out * out_dw[wo + k];
      for (uint32_t k = 0; k < kHiddenDim; ++k) combined_db[bo + k] += lr_out * out_db[bo + k];
    }
    {
      size_t wo = weight_offsets_[4], bo = bias_offsets_[4];
      for (uint32_t t = 0; t < kHiddenDim; ++t) {
        size_t idx = wo + target_out * kHiddenDim + t;
        combined_dw[idx] += lr_out * out_dw[idx];
      }
      combined_db[bo + target_out] += lr_out * out_db[bo + target_out];
    }
  }  // per target_out

  // ---- Trust-region step, anti-flip damping, EMA smoothing, weight update
  // (nn_gpu.cu's Steps 5-9) ----
  constexpr float kEmaDecay = 0.85f;
  constexpr float kTrustK = 0.08f;
  constexpr float kMaxStep = 0.02f;
  constexpr float kMinStep = 1e-4f;
  constexpr float kAntiFlipDamp = 0.5f;
  constexpr float kWClamp = 10.0f;

  // Normalize the combined gradient to a unit direction (trust-region:
  // decouple update direction from magnitude).
  double g_norm_sq = 0.0;
  for (float v : combined_dw) g_norm_sq += static_cast<double>(v) * v;
  for (float v : combined_db) g_norm_sq += static_cast<double>(v) * v;
  float g_norm = static_cast<float>(std::sqrt(g_norm_sq)) + 1e-8f;
  float inv_norm = 1.0f / g_norm;
  for (float& v : combined_dw) v *= inv_norm;
  for (float& v : combined_db) v *= inv_norm;

  // Step size = k * average |raw error| across active outputs/samples,
  // clamped to [MIN_STEP, MAX_STEP].
  float sum_err = 0.0f;
  int err_count = 0;
  for (size_t si = 0; si < num_samples; ++si) {
    for (int o = 0; o < static_cast<int>(kOutputDim); ++o) {
      float e = std::fabs(d5_raw[si][o]);
      if (e > 0.0f) { sum_err += e; ++err_count; }
    }
  }
  float avg_err = (err_count > 0) ? sum_err / static_cast<float>(err_count) : 0.0f;
  float step = std::max(kMinStep, std::min(kMaxStep, kTrustK * avg_err));

  // Anti-flip damping: if the new gradient direction points opposite the
  // EMA (recent) direction, halve the step -- only after warmup, since the
  // first few calls have no meaningful EMA yet.
  bool warmed_up = (sgd_call_count_ > 3);
  if (warmed_up) {
    // TRUNK ONLY -- W5/b5 are excluded. NeuroPress accumulates this dot over
    // DW1..DB4 and stops (nn_gpu.cu:1380-1402); the trust-region norm just
    // above it (:1319-1326) deliberately does include W5/b5, so the
    // narrower scope here is intentional upstream, not an oversight.
    // Including the unprojected, sign-stable W5 gradient biases the dot
    // positive and damps the step less often than upstream.
    const size_t dw_trunk_end = weight_offsets_[4];
    const size_t db_trunk_end = bias_offsets_[4];
    double dot_ema = 0.0;
    for (size_t i = 0; i < dw_trunk_end; ++i)
      dot_ema += static_cast<double>(combined_dw[i]) * ema_weights_[i];
    for (size_t i = 0; i < db_trunk_end; ++i)
      dot_ema += static_cast<double>(combined_db[i]) * ema_biases_[i];
    if (dot_ema < 0.0) step *= kAntiFlipDamp;
  }

  // Sanitize (a corrupted forward pass could have produced NaN/Inf) before
  // it permanently poisons the EMA buffer.
  for (float& v : combined_dw) if (!std::isfinite(v)) v = 0.0f;
  for (float& v : combined_db) if (!std::isfinite(v)) v = 0.0f;

  float ema_new = 1.0f - kEmaDecay;
  for (size_t i = 0; i < ema_weights_.size(); ++i)
    ema_weights_[i] = kEmaDecay * ema_weights_[i] + ema_new * combined_dw[i];
  for (size_t i = 0; i < ema_biases_.size(); ++i)
    ema_biases_[i] = kEmaDecay * ema_biases_[i] + ema_new * combined_db[i];

  ++sgd_call_count_;

  // NaN/Inf guard: a single corrupted sample must not be allowed to destroy
  // the network. The EMA above is already sanitized, so skipping the
  // update here just means "wait for a cleaner gradient next call."
  if (!std::isfinite(step) || !std::isfinite(g_norm)) {
    return false;
  }

  // Output 1 (decompression time) is NOT this pass's to update: its head
  // weights belong to TrainDecompHead(), fed by real measured times only a
  // later read can supply. Mirrors nnSGDKernel's `if (out == 1) continue;`
  // and `t != 1` (nn_gpu.cu), which skip exactly these weights for exactly
  // this reason. Note upstream still lets output 1's error flow into the
  // shared trunk (W1-W4) -- only the head itself is withheld -- so the skip
  // belongs here at the update, not earlier in the gradient accumulation.
  // Without it the two mechanisms would fight over w5 row 1 as soon as a
  // real decompression time reaches Train().
  const size_t kSkipW5RowBegin = weight_offsets_[4] + 1 * kHiddenDim;
  const size_t kSkipW5RowEnd = kSkipW5RowBegin + kHiddenDim;
  const size_t kSkipB5Idx = bias_offsets_[4] + 1;

  for (size_t i = 0; i < weights_.size(); ++i) {
    if (i >= kSkipW5RowBegin && i < kSkipW5RowEnd) continue;
    float w = weights_[i] - step * ema_weights_[i];
    weights_[i] = std::max(-kWClamp, std::min(kWClamp, w));
  }
  for (size_t i = 0; i < biases_.size(); ++i) {
    if (i == kSkipB5Idx) continue;
    float b = biases_[i] - step * ema_biases_[i];
    biases_[i] = std::max(-kWClamp, std::min(kWClamp, b));
  }
  return true;
}

const std::vector<float>& NeuroPressNNPredictor::DebugWeights() {
#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    gpu::NeuroPressGpuDownloadWeights(gpu_weights_.get(), weights_.data(),
                                      biases_.data());
  }
#endif
  return weights_;
}

const std::vector<float>& NeuroPressNNPredictor::DebugBiases() {
#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    gpu::NeuroPressGpuDownloadWeights(gpu_weights_.get(), weights_.data(),
                                      biases_.data());
  }
#endif
  return biases_;
}

bool NeuroPressNNPredictor::TrainDecompHead(
    const std::vector<CompressionFeatures>& features,
    const std::vector<double>& decompression_times_ms) {
  if (!is_ready_ || features.empty() ||
      features.size() != decompression_times_ms.size()) {
    return false;
  }

  // Head-only constants, distinct from Train()'s -- see nn_gpu.cu's
  // nnBatchedDecompSGDKernel.
  constexpr float kDecompTrustK = 0.15f;
  constexpr float kDecompMaxStep = 0.05f;
  constexpr float kDecompMinStep = 1e-4f;
  constexpr float kDecompWClamp = 5.0f;
  constexpr float kDecompErrClamp = 2.0f;   // log-space error clamp
  constexpr float kDecompNoiseGate = 0.05f;

  const size_t w5_row1 = weight_offsets_[4] + 1 * kHiddenDim;
  const size_t b5_idx1 = bias_offsets_[4] + 1;

#if CTP_ENABLE_NEUROPRESS_GPU
  // When inference runs on the device, the device copy is the live one --
  // updating only the host vectors would leave the change invisible to every
  // subsequent Predict(). Pull the current weights down first, do the update
  // on the host (one implementation of this ported math, not two), and push
  // the result back below.
  if (gpu_weights_) {
    gpu::NeuroPressGpuDownloadWeights(gpu_weights_.get(), weights_.data(),
                                      biases_.data());
  }
#endif

  std::vector<float> acc_gw(kHiddenDim, 0.0f);
  float acc_gb = 0.0f;
  float acc_abs_err = 0.0f;
  int valid = 0;

  for (size_t si = 0; si < features.size(); ++si) {
    double measured = decompression_times_ms[si];
    if (measured <= 0.0) continue;  // not measured -- nothing to learn from

    // Forward through the frozen trunk to h4, then output 1's head only.
    auto x = FeaturesTo8Input(features[si]);
    auto x_norm = Standardize(x);
    SGDActivations a = ForwardWithCache(x_norm);

    float y_norm = biases_[b5_idx1];
    for (uint32_t i = 0; i < kHiddenDim; ++i) {
      y_norm += weights_[w5_row1 + i] * a.h4[i];
    }

    float y_std1 = std::max(y_stds_[1], 1e-8f);
    float pred_log = y_norm * y_std1 + y_means_[1];

    float clamped =
        std::max(0.01f, std::min(static_cast<float>(measured), 5000.0f));
    float target_log = std::log1p(clamped);

    float err_log = pred_log - target_log;
    err_log = std::max(-kDecompErrClamp, std::min(kDecompErrClamp, err_log));
    float err = err_log / y_std1;

    // Noise gate: a sub-threshold error is timing jitter, not a real miss.
    if (std::fabs(err) < kDecompNoiseGate) continue;

    for (uint32_t t = 0; t < kHiddenDim; ++t) {
      acc_gw[t] += err * a.h4[t];
    }
    acc_gb += err;
    acc_abs_err += std::fabs(err);
    ++valid;
  }

  if (valid == 0) return false;

  const float inv_n = 1.0f / static_cast<float>(valid);
  for (float& g : acc_gw) g *= inv_n;
  acc_gb *= inv_n;

  // TWO norms, mirroring an upstream quirk that is easy to miss and that a
  // differential test caught immediately (nn_gpu.cu):
  //
  //   float g_norm = sqrtf(s_reduce[0] + (t == 0 ? acc_gb*acc_gb : 0.0f)) + 1e-8f;
  //
  // g_norm is computed PER THREAD, so the bias term is only present on
  // thread 0. Thread 0 owns w5[row1][0] and b5[1]; threads 1..63 own
  // w5[row1][1..63] and normalize WITHOUT the bias term. Collapsing this to
  // one norm shifts most of the row by ~0.25%, which is exactly what the
  // parity harness reported before this was replicated.
  double norm_sq_w = 0.0;
  for (float g : acc_gw) norm_sq_w += static_cast<double>(g) * g;
  const float g_norm_w_only =
      static_cast<float>(std::sqrt(norm_sq_w)) + 1e-8f;
  const float g_norm_with_bias =
      static_cast<float>(
          std::sqrt(norm_sq_w + static_cast<double>(acc_gb) * acc_gb)) +
      1e-8f;
  const float g_norm = g_norm_with_bias;  // used for the finiteness guard

  // Trust region proportional to the error: a well-calibrated head takes
  // small steps, which is what stops it oscillating.
  float mean_abs_err = acc_abs_err * inv_n;
  float step = std::max(kDecompMinStep,
                        std::min(kDecompMaxStep, kDecompTrustK * mean_abs_err));

  if (!std::isfinite(step) || !std::isfinite(g_norm)) return false;

  for (uint32_t t = 0; t < kHiddenDim; ++t) {
    // Thread 0's norm carries the bias term; every other thread's does not.
    float gw_normed = acc_gw[t] / ((t == 0) ? g_norm_with_bias : g_norm_w_only);
    if (!std::isfinite(gw_normed)) continue;
    float w = weights_[w5_row1 + t] - step * gw_normed;
    weights_[w5_row1 + t] =
        std::max(-kDecompWClamp, std::min(kDecompWClamp, w));
  }
  float gb_normed = acc_gb / g_norm_with_bias;  // thread 0
  if (std::isfinite(gb_normed)) {
    float b = biases_[b5_idx1] - step * gb_normed;
    biases_[b5_idx1] = std::max(-kDecompWClamp, std::min(kDecompWClamp, b));
  }

#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    gpu::NeuroPressGpuUploadWeights(gpu_weights_.get(), weights_.data(),
                                    biases_.data());
  }
#endif
  return true;
}

}  // namespace ctp::compress::model
