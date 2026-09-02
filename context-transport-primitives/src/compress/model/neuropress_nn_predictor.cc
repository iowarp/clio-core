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

  // Accept v1 as well as v2. Upstream reads both (`if (version != 1 &&
  // version != 2)` -- nn_gpu.cu) and differs only in the trailing
  // feature bounds, which v1 files do not carry. Rejecting v1 meant Clio
  // could not load a weight file the original project loads fine.
  if (!file || magic != 0x4E4E5754 || (version != 1 && version != 2)) {
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

  // Feature bounds are a v2 addition. A v1 file simply ends here, and
  // upstream fills in wide-open defaults rather than failing
  // (nn_gpu.cu). Nothing consults them at inference or training; they are
  // parsed so a v2 file round-trips through Save().
  x_mins_.resize(input_dim);
  x_maxs_.resize(input_dim);
  if (version >= 2) {
    file.read(reinterpret_cast<char*>(x_mins_.data()), input_dim * 4);
    file.read(reinterpret_cast<char*>(x_maxs_.data()), input_dim * 4);
    if (!file) {
      return false;
    }
  } else {
    for (uint32_t i = 0; i < input_dim; ++i) {
      x_mins_[i] = -1e30f;
      x_maxs_[i] = 1e30f;
    }
  }

  // Online-learning state (log-variance, EMA gradient, call count) is
  // device-resident and reset by NeuroPressGpuLoad below, matching upstream's
  // per-process state -- a reload restarts learning there too.

#if CTP_ENABLE_NEUROPRESS_GPU
  // Upload to device -- this is what actually makes Predict()/Train() run
  // as CUDA kernels below, matching NeuroPress's own device-resident design.
  gpu::NeuroPressGpuWeights* raw = gpu::NeuroPressGpuLoad(
      weights_.data(), weights_.size(), biases_.data(), biases_.size(),
      x_means_.data(), x_stds_.data(), y_means_.data(), y_stds_.data());
  gpu_weights_.reset(raw, [](gpu::NeuroPressGpuWeights* p) {
    gpu::NeuroPressGpuFree(p);
  });
  if (!raw) {
    // HALT, do not degrade. This is a build with GPU support whose device was
    // unavailable at load time. Upstream has no CPU implementation of this
    // network at all and ends the call on any such failure
    // (GPUCOMPRESS_ERROR_NN_NOT_LOADED, gpucompress_compress.cpp and
    // :285), so answering from the host port instead would be NeuroPress
    // running somewhere NeuroPress never runs.
    //
    // Failing the load is what makes that impossible rather than merely
    // discouraged: IsReady() stays false, so the runtime's
    // `neuropress_predictor_->IsReady()` gate (compressor_runtime.cc) never
    // opens and the model is simply unavailable -- the same outcome as
    // upstream's gpucompress_init() failing on a host with no device. What
    // the runtime does next (legacy heuristics) is a different MODEL, chosen
    // visibly, not this model relocated to the CPU.
    std::fprintf(stderr,
                 "clio ERROR: NeuroPress GPU weights could not be created on a "
                 "CUDA-enabled build; refusing to load. NeuroPress has no CPU "
                 "path -- it will not be used rather than run on the host.\n");
    weights_.clear();
    biases_.clear();
    is_ready_ = false;
    return false;
  }
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

std::vector<float> NeuroPressNNPredictor::FeaturesTo8Input(
    const CompressionFeatures& features, bool apply_lossless_sentinel) const {
  // NeuroPress expects: [algo_id, quant, shuffle, error_bound, data_size,
  // entropy, mad, second_derivative]
  int base_id = static_cast<int>(features.library_config_id) / 10;
  float algo_id = static_cast<float>(NeuroPressAlgoIdForBaseId(base_id));

  // Lossless configs carry a SENTINEL error bound of 1e-7, not 0. That is
  // how the model was trained -- neural_net/core/configs.py:44,
  // `eb_val = eb if quant else 1e-7` -- and nn_gpu.cu repeats it at
  // inference with the comment "training used 1e-7 sentinel for lossless
  // (quant==0). Inference must match -- do not pass raw 0.0 for lossless
  // configs." Passing 0.0 put input 3 about 2.4e-6 standard deviations off
  // upstream's value for every lossless candidate, which is every candidate
  // Clio ranks.
  // Sentinel for INFERENCE only -- see the header. Training feeds the raw
  // bound on both SGD paths, so a lossless config contributes 0.0 there.
  const float error_bound_enc =
      (apply_lossless_sentinel && !features.quantize)
          ? 1e-7f
          : static_cast<float>(features.error_bound);

  return {algo_id,
          static_cast<float>(features.quantize),      // quant preprocessor
          static_cast<float>(features.byte_shuffle),  // shuffle preprocessor
          error_bound_enc,                            // lossy/quant bound
          static_cast<float>(features.chunk_size_bytes),
          static_cast<float>(features.shannon_entropy),
          static_cast<float>(features.mad),
          static_cast<float>(features.second_derivative_mean)};
}


CompressionPrediction NeuroPressNNPredictor::Predict(
    const CompressionFeatures& features) {
  if (!is_ready_) {
    return CompressionPrediction();
  }
  auto batch = PredictBatch({features});
  return batch.empty() ? CompressionPrediction() : batch.front();
}

bool NeuroPressNNPredictor::PredictBatchFull(
    const std::vector<CompressionFeatures>& batch, FullOutputs* out) const {
  if (out == nullptr || batch.empty()) return false;
#if CTP_ENABLE_NEUROPRESS_GPU
  if (!gpu_weights_) return false;

  // Same input assembly as PredictBatch, including the lossless sentinel --
  // asking the two paths different questions would make any comparison
  // between them meaningless.
  std::vector<float> raw_inputs(batch.size() * kInputDim);
  for (size_t i = 0; i < batch.size(); ++i) {
    auto x = FeaturesTo8Input(batch[i]);
    std::copy(x.begin(), x.end(),
              raw_inputs.begin() + static_cast<long>(i * kInputDim));
  }

  const size_t n = batch.size();
  out->comp_time_ms.assign(n, 0.0f);
  out->decomp_time_ms.assign(n, 0.0f);
  out->ratio.assign(n, 0.0f);
  out->psnr_db.assign(n, 0.0f);
  out->rmse.assign(n, 0.0f);
  out->max_error.assign(n, 0.0f);
  out->mae.assign(n, 0.0f);
  out->ssim.assign(n, 0.0f);

  std::lock_guard<std::mutex> lock(*model_mutex_);
  if (!gpu::NeuroPressGpuInferBatchFull(
          gpu_weights_.get(), raw_inputs.data(), static_cast<int>(n),
          out->comp_time_ms.data(), out->decomp_time_ms.data(),
          out->ratio.data(), out->psnr_db.data(), out->rmse.data(),
          out->max_error.data(), out->mae.data(), out->ssim.data())) {
    *out = FullOutputs{};
    return false;
  }
  return true;
#else
  (void)batch;
  return false;
#endif
}

std::vector<CompressionPrediction> NeuroPressNNPredictor::PredictBatch(
    const std::vector<CompressionFeatures>& batch) {
  if (batch.empty()) {
    return {};
  }
  // Readers take the same lock as the writers below. Upstream serializes
  // this with a completion event instead (cudaStreamWaitEvent on g_sgd_done,
  // nn_gpu.cu), which Clio cannot use because TrainDecompHead's update
  // is host-side, not a kernel on the SGD stream.
  std::lock_guard<std::mutex> model_lock(*model_mutex_);
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
      // Already clamped inside the kernel (nn_gpu.cu order: sanity
      // ceiling first, then the policy floors/caps). Repeat the policy half
      // here so the CPU and GPU paths read identically.
      //
      // Upstream's literal 100, not the configurable cap: this entry point
      // takes no RankingWeights and therefore has no cost model, so there is
      // no cap to honour. Callers that raise the ceiling go through
      // PredictBatchDeviceStats, which carries the weights.
      results.emplace_back(
          std::max(0.1, std::min(100.0, static_cast<double>(ratio[i]))),
          std::max(0.0, std::min(120.0, static_cast<double>(psnr[i]))),
          std::max(1.0, static_cast<double>(comp_time[i])),
          std::max(1.0, static_cast<double>(decomp_time[i])), infer_ms);
    }
    return results;
  }
  // Unreachable: Load() refuses on this build when the device weights could
  // not be created, so a ready predictor always has them. Kept as a hard stop
  // rather than a comment, because falling through to the host loop below is
  // exactly the behaviour that must not exist -- upstream has no CPU network
  // and ends the call instead (gpucompress_compress.cpp).
  return {};
#else
  // No CPU path -- see TrainDecompHead.
  (void)batch;
  return {};
#endif  // !CTP_ENABLE_NEUROPRESS_GPU
}

std::vector<CompressionPrediction>
NeuroPressNNPredictor::PredictBatchDeviceStats(
    const void* device_stats, const std::vector<CompressionFeatures>& batch,
    void* stream, const RankingWeights* weights, std::vector<int>* out_order,
    double min_psnr, std::vector<double>* out_scores, FullOutputs* out_full,
    const ctp::compress::preprocess::PredictionReuseContext* reuse,
    ctp::compress::preprocess::PredictionReuseOutcome* out_outcome) {
#if CTP_ENABLE_NEUROPRESS_GPU
  // Same lock PredictBatch takes, for the same reason: TrainDecompHead()
  // downloads every parameter, edits two and uploads them all back, so an
  // inference reading the device weights mid-update would see a torn model.
  // Upstream serializes the same hazard with cudaStreamWaitEvent on
  // g_sgd_done (nn_gpu.cu). Omitting it here would have left this
  // path unprotected while the one beside it was guarded -- learning is off
  // in the configuration this path was built for, which is exactly what
  // would have kept the gap invisible.
  std::lock_guard<std::mutex> model_lock(*model_mutex_);
  if (!is_ready_ || !gpu_weights_ || !device_stats || batch.empty()) return {};

  auto start_time = std::chrono::high_resolution_clock::now();

  // Only the four genuinely host-side inputs are assembled here. Inputs 5-7
  // (entropy, MAD, second derivative) are deliberately NOT read off `batch`
  // -- they live on the device and the kernel reads them there. Input 4 is
  // the chunk size, identical for every candidate, so it rides as a scalar.
  std::vector<float> desc(batch.size() * 4);
  for (size_t i = 0; i < batch.size(); ++i) {
    // RAW error bound, sentinel NOT applied. Upstream substitutes the 1e-7
    // lossless sentinel inside its inference kernel
    // (`input_raw[3] = (quant == 0) ? 1e-7f : eb_enc`, nn_gpu.cu) rather
    // than in whatever assembled the inputs, so InferKernelDeviceStats does
    // it there too -- the descriptor now carries what the caller actually
    // configured and the substitution is the model's own, as it is upstream.
    const auto x = FeaturesTo8Input(batch[i], /*apply_lossless_sentinel=*/false);
    desc[i * 4 + 0] = x[0];  // algo_id
    desc[i * 4 + 1] = x[1];  // quantize
    desc[i * 4 + 2] = x[2];  // byte_shuffle
    desc[i * 4 + 3] = x[3];  // error bound, raw
  }

  std::vector<float> comp_time(batch.size()), decomp_time(batch.size()),
      ratio(batch.size()), psnr(batch.size());

  // When the caller wants the ordering too, it is computed by the same kernel
  // launch sequence rather than on the host -- see RankKernel. `out_order`
  // being null keeps this a pure inference call.
  // Upstream's ACTION INDICES, one per candidate. That is all the kernel
  // needs: it decodes each one back into algo/quantize/shuffle itself, the
  // way upstream decodes its thread index (nn_gpu.cu), so nothing
  // about a configuration is assembled here any more. The error bound rides
  // in as a scalar and the lossless sentinel is applied in-kernel.
  std::vector<int> action_ids(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    const int base_id = static_cast<int>(batch[i].library_config_id) / 10;
    action_ids[i] = NeuroPressActionId(base_id, batch[i].quantize != 0,
                                       batch[i].byte_shuffle != 0);
  }

  // The chunk's bound: candidates carry it only when they quantize, so the
  // maximum recovers it (see the mask note below).
  double chunk_error_bound_for_kernel = 0.0;
  for (const auto& f : batch) {
    chunk_error_bound_for_kernel =
        std::max(chunk_error_bound_for_kernel, f.error_bound);
  }

  gpu::GpuRankParams rank;
  const gpu::GpuRankParams *rank_ptr = nullptr;
  int *order_ptr = nullptr;
  double *scores_ptr = nullptr;
  if (out_order != nullptr && weights != nullptr && weights->use_cost_model) {
    rank.data_size_bytes = batch[0].chunk_size_bytes;
    rank.w_compress_time = weights->w_cost_compress_time;
    rank.w_decompress_time = weights->w_cost_decompress_time;
    rank.w_io = weights->w_cost_io;
    rank.bandwidth_bytes_per_ms = weights->bandwidth_bytes_per_ms;
    rank.ratio_cap = weights->ratio_cap;
    // The two mask inputs.
    //
    // The bound is the CHUNK's, i.e. upstream's cfg.error_bound -- NOT
    // batch[0]'s. Candidates carry the bound only when they quantize
    // (`c.error_bound = quant ? error_bound : 0.0`, neuropress_bridge.cc), and
    // candidate 0 is always the unquantized one, so reading it there reported
    // 0 on every run and masked the quantize half even when the caller had
    // asked for a lossy write. Taking the maximum recovers the configured
    // bound whenever any quantize candidate exists, and yields 0 when none
    // does -- in which case the mask is moot, since nothing is quantized.
    rank.error_bound = chunk_error_bound_for_kernel;
    rank.min_psnr = min_psnr;
    out_order->assign(batch.size(), 0);
    rank_ptr = &rank;
    order_ptr = out_order->data();
    if (out_scores != nullptr) {
      out_scores->assign(batch.size(), 0.0);
      scores_ptr = out_scores->data();
    }
  }

  // Outputs 4-7, only when asked. Sized here so the pointers below are valid
  // for the whole call; left empty (and null) otherwise, which is what tells
  // the kernel to skip the transforms and the fetch to stop after output 3.
  const size_t nb = batch.size();
  if (out_full != nullptr) {
    out_full->comp_time_ms.assign(nb, 0.0f);
    out_full->decomp_time_ms.assign(nb, 0.0f);
    out_full->ratio.assign(nb, 0.0f);
    out_full->psnr_db.assign(nb, 0.0f);
    out_full->rmse.assign(nb, 0.0f);
    out_full->max_error.assign(nb, 0.0f);
    out_full->mae.assign(nb, 0.0f);
    out_full->ssim.assign(nb, 0.0f);
  }
  float *q_rmse = out_full ? out_full->rmse.data() : nullptr;
  float *q_maxe = out_full ? out_full->max_error.data() : nullptr;
  float *q_mae = out_full ? out_full->mae.data() : nullptr;
  float *q_ssim = out_full ? out_full->ssim.data() : nullptr;

  if (!gpu::NeuroPressGpuInferBatchDeviceStats(
          gpu_weights_.get(), device_stats, action_ids.data(),
          static_cast<int>(batch.size()),
          static_cast<float>(batch[0].chunk_size_bytes),
          static_cast<float>(chunk_error_bound_for_kernel), stream,
          comp_time.data(), decomp_time.data(), ratio.data(), psnr.data(),
          rank_ptr, order_ptr, scores_ptr, q_rmse, q_maxe, q_mae, q_ssim,
          reuse, out_outcome)) {
    if (out_full != nullptr) *out_full = FullOutputs{};
    if (out_order != nullptr) out_order->clear();
    if (out_scores != nullptr) out_scores->clear();
    // Same contract as PredictBatch: an empty vector, never a zero-filled
    // one that would read as a complete and pessimistic ranking.
    return {};
  }
  if (out_full != nullptr) {
    out_full->comp_time_ms.assign(comp_time.begin(), comp_time.end());
    out_full->decomp_time_ms.assign(decomp_time.begin(), decomp_time.end());
    out_full->ratio.assign(ratio.begin(), ratio.end());
    out_full->psnr_db.assign(psnr.begin(), psnr.end());
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  const double infer_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count() /
      static_cast<double>(batch.size());

  const double kRatioCap = (weights != nullptr) ? weights->ratio_cap : 100.0;
  std::vector<CompressionPrediction> results;
  results.reserve(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    // Identical post-clamps to PredictBatch's GPU branch, so a chunk routed
    // through either path lands on the same numbers -- and with the SAME cap
    // the kernel used, or this re-clamp silently undoes a raised ceiling.
    results.emplace_back(
        std::max(0.1, std::min(kRatioCap, static_cast<double>(ratio[i]))),
        std::max(0.0, std::min(120.0, static_cast<double>(psnr[i]))),
        std::max(1.0, static_cast<double>(comp_time[i])),
        std::max(1.0, static_cast<double>(decomp_time[i])), infer_ms);
  }
  return results;
#else
  (void)device_stats;
  (void)batch;
  (void)stream;
  (void)weights;
  (void)min_psnr;
  if (out_order != nullptr) out_order->clear();
  if (out_scores != nullptr) out_scores->clear();
  return {};
#endif
}

bool NeuroPressNNPredictor::Train(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels) {
  return TrainDeviceStats(features, labels, nullptr);
}

bool NeuroPressNNPredictor::TrainDeviceStats(
    const std::vector<CompressionFeatures>& features,
    const std::vector<TrainingLabels>& labels, const void* device_stats) {
  if (!is_ready_ || features.empty() || features.size() != labels.size()) {
    return false;
  }
  // Matches NeuroPress's g_sgd_mutex around every SGD dispatch
  // (gpucompress_compress.cpp, :1021, gpucompress_learning.cpp).
  std::lock_guard<std::mutex> model_lock(*model_mutex_);
  const size_t num_samples =
      std::min(features.size(), static_cast<size_t>(kMaxSGDSamples));

#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    // Real SGD kernel: forward + backward + weight update entirely
    // on-device (see neuropress_nn_gpu_kernels.cu's SGDKernel), mirroring
    // nnSGDKernel exactly rather than the CPU port below.
    std::vector<gpu::NeuroPressGpuSGDSample> gpu_samples(num_samples);
    for (size_t si = 0; si < num_samples; ++si) {
      auto x = FeaturesTo8Input(features[si], /*sentinel=*/false);
      std::copy(x.begin(), x.end(), gpu_samples[si].raw_input);
      gpu_samples[si].actual_ratio = labels[si].compression_ratio;
      gpu_samples[si].actual_comp_time_ms = labels[si].compression_time_ms;
      gpu_samples[si].actual_decomp_time_ms = labels[si].decompression_time_ms;
      gpu_samples[si].actual_psnr_db = labels[si].psnr_db;
    }
    return gpu::NeuroPressGpuTrain(gpu_weights_.get(), gpu_samples.data(),
                                   static_cast<int>(num_samples),
                                   learning_rate_, device_stats);
  }
  // Unreachable for the same reason as PredictBatch's: Load() refuses without
  // device weights. Refuse the update rather than running SGD on the host --
  // upstream's online learning is a device kernel and nothing else.
  return false;
#else
  // No CPU path -- see TrainDecompHead.
  (void)features;
  (void)labels;
  (void)device_stats;
  return false;
#endif  // !CTP_ENABLE_NEUROPRESS_GPU
}

int NeuroPressNNPredictor::DebugSgdCallCount() {
#if CTP_ENABLE_NEUROPRESS_GPU
  if (gpu_weights_) {
    return gpu::NeuroPressGpuSgdCallCount(gpu_weights_.get());
  }
#endif
  return 0;
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
  // Critical here, not merely tidy: the update below is a host-side
  // read-modify-write that downloads every parameter, edits w5 row 1 and
  // b5[1], and uploads them all back. Without the lock a concurrent Train()
  // is silently reverted, because the upload carries a pre-Train snapshot of
  // the whole trunk. Upstream cannot lose an update this way -- its decomp
  // update is a device kernel that writes only those two slots
  // (nn_gpu.cu).
  std::lock_guard<std::mutex> model_lock(*model_mutex_);

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
  // Device path: one kernel that touches only w5 row 1 and b5[1], as upstream
  // does (nnBatchedDecompSGDKernel, nn_gpu.cu). What this replaces is a
  // host read-modify-write of the WHOLE parameter set -- download all, edit
  // two, upload all -- which is both a device kernel's work done on the CPU
  // and the reason a concurrent Train() could be silently reverted: the upload
  // carried a pre-Train snapshot of the entire trunk. Writing two slots in
  // place removes that hazard rather than locking around it.
  if (gpu_weights_) {
    std::vector<gpu::NeuroPressGpuDecompSample> samples;
    samples.reserve(features.size());
    for (size_t si = 0; si < features.size(); ++si) {
      const double measured = decompression_times_ms[si];
      if (measured <= 0.0) continue;  // not measured -- nothing to learn from
      // RAW bound: the sentinel is inference-only (nn_gpu.cu).
      const auto x = FeaturesTo8Input(features[si],
                                      /*apply_lossless_sentinel=*/false);
      const int base_id =
          static_cast<int>(features[si].library_config_id) / 10;
      gpu::NeuroPressGpuDecompSample s{};
      s.action = NeuroPressActionId(base_id, features[si].quantize != 0,
                                    features[si].byte_shuffle != 0);
      s.error_bound_enc = x[3];
      s.data_size_enc = x[4];
      s.entropy = x[5];
      s.mad = x[6];
      s.second_derivative = x[7];
      s.actual_decomp_ms = static_cast<float>(measured);
      samples.push_back(s);
    }
    if (samples.empty()) return false;
    return gpu::NeuroPressGpuTrainDecompHead(
        gpu_weights_.get(), samples.data(), static_cast<int>(samples.size()));
  }
  // Unreachable on a GPU build: Load() refuses without device weights.
  return false;
#else
  // No CPU path. NeuroPress's decompression-head update exists only as a
  // CUDA kernel (nnBatchedDecompSGDKernel), and the project cannot even be
  // configured without CUDA -- project(... LANGUAGES CXX CUDA C) with
  // find_package(CUDAToolkit REQUIRED). A host reimplementation would be
  // Clio inventing behaviour upstream does not have.
  (void)features;
  (void)decompression_times_ms;
  return false;
#endif  // !CTP_ENABLE_NEUROPRESS_GPU
}

}  // namespace ctp::compress::model
