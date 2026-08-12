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

// Copyright 2024 IOWarp contributors
#include <clio_cte/compressor/compressor_runtime.h>

#include <clio_ctp/serialize/msgpack_wrapper.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <string>
#include <tuple>
#include <vector>

#include "clio_runtime/work_orchestrator.h"
#include "clio_runtime/worker.h"
#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/data_stats.h"
#include "clio_ctp/compress/preprocess/byte_shuffle.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/compress/model/ranking.h"
#include "clio_ctp/util/logging.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"

namespace clio::cte::compressor {

// Bring chi namespace items into scope for CLIO_CUR_WORKER macro
using clio::run::chi_cur_worker_key_;
using clio::run::Worker;

/**
 * Compression header prepended to compressed data for self-describing format.
 * This allows decompression without external metadata.
 */
/**
 * Byte-shuffle element size packed into the free high bits of
 * compress_preset_.
 *
 * NeuroPress's action space is algorithm x quantize x byte-shuffle
 * (internal.hpp's decodeAction), and with error_bound=0 its own ranking masks
 * every quantize action to -INFINITY -- so byte-shuffle is the ONLY extra
 * dimension a lossless deployment can reach, and Clio not applying it made
 * half of upstream's reachable actions unreachable here (the parity harness
 * shows native selecting actions 20/21/23, all shuffle=1).
 *
 * Packed rather than added as a field because CompressionHeader is a fixed
 * 24 bytes by static_assert and is the on-disk format: growing it would make
 * every already-compressed blob unreadable. compress_preset_ only ever holds
 * 0-3, so the upper bits are free, and a blob written before this change
 * decodes to elem_size 0 -- "not shuffled" -- which is exactly right.
 */
constexpr uint32_t kPresetMask = 0xFFu;
constexpr uint32_t kShuffleShift = 8;
constexpr uint32_t kShuffleMask = 0xFFu;

inline uint32_t PackPreset(uint32_t preset, uint32_t shuffle_elem_size) {
  return (preset & kPresetMask) |
         ((shuffle_elem_size & kShuffleMask) << kShuffleShift);
}
inline uint32_t UnpackPreset(uint32_t packed) { return packed & kPresetMask; }
inline uint32_t UnpackShuffle(uint32_t packed) {
  return (packed >> kShuffleShift) & kShuffleMask;
}

struct CompressionHeader {
  static constexpr uint32_t kMagic = 0x43544543;  // "CTEC" in ASCII
  uint32_t magic_;            // Magic number to identify compressed data
  uint32_t compress_lib_;     // Compression library ID
  uint32_t compress_preset_;  // Compression preset | shuffle elem size << 8
  uint64_t original_size_;    // Original uncompressed size

  CompressionHeader()
      : magic_(kMagic),
        compress_lib_(0),
        compress_preset_(0),
        original_size_(0) {}

  CompressionHeader(uint32_t lib, uint32_t preset, uint64_t orig_size)
      : magic_(kMagic),
        compress_lib_(lib),
        compress_preset_(preset),
        original_size_(orig_size) {}

  bool IsValid() const { return magic_ == kMagic; }
};
static_assert(sizeof(CompressionHeader) == 24,
              "CompressionHeader must be 24 bytes");

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Load configuration from compose YAML (or direct CreateParams)
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target

  // Initialize the core client using next_pool_id from compose
  if (!config_.next_pool_id_.IsNull()) {
    core_client_ = std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }

  // Initialize atomic counters
  compression_logical_time_ = 0;

  // tag_consumers_ is lazily populated by RegisterConsumer; nothing to
  // preallocate here. The map is empty when tracking_enabled_=false.

  // Seed previous CPU times so PollNodeLoad's first delta is well-defined.
  prev_cpu_times_ = ctp::SystemInfo::GetCpuTimes();

  // Load Q-table model if configured (primary prediction method)
  if (!config_.qtable_model_path_.empty()) {
    try {
      HLOG(kDebug, "Loading Q-table model from: {}",
           config_.qtable_model_path_);
      qtable_predictor_ = std::make_unique<QTablePredictor>();
      if (qtable_predictor_->Load(config_.qtable_model_path_)) {
        HLOG(kDebug, "Q-table model loaded successfully with {} states",
             qtable_predictor_->GetNumStates());
      } else {
        HLOG(kWarning, "Failed to load Q-table model from: {}",
             config_.qtable_model_path_);
        qtable_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading Q-table model: {}", e.what());
      qtable_predictor_.reset();
    }
  }

  // Load LinReg table model if configured
  if (!config_.linreg_model_path_.empty()) {
    try {
      HLOG(kDebug, "Loading LinReg table model from: {}",
           config_.linreg_model_path_);
      linreg_predictor_ = std::make_unique<LinRegTablePredictor>();
      if (linreg_predictor_->Load(config_.linreg_model_path_)) {
        HLOG(kDebug, "LinReg table model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load LinReg table model from: {}",
             config_.linreg_model_path_);
        linreg_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading LinReg table model: {}", e.what());
      linreg_predictor_.reset();
    }
  }

  // Load distribution classifier if configured
  if (!config_.distribution_model_path_.empty()) {
    // Note: DistributionClassifier is template-based - use
    // DistributionClassifierFactory::Classify() directly No model loading
    // needed - the factory uses built-in mathematical classification
    HLOG(kDebug,
         "Distribution classifier available via factory (no model loading "
         "required)");
  }

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
  // Load DNN model weights as fallback if Q-table not available
  if (!qtable_predictor_ && !config_.dnn_model_weights_path_.empty()) {
    try {
      HLOG(kDebug, "Loading DNN model weights from: {}",
           config_.dnn_model_weights_path_);
      nn_predictor_ = std::make_unique<DenseNNPredictor>();
      if (nn_predictor_->LoadWeights(config_.dnn_model_weights_path_)) {
        HLOG(kDebug, "DNN model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load DNN model weights from: {}",
             config_.dnn_model_weights_path_);
        nn_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading DNN model: {}", e.what());
      nn_predictor_.reset();
    }
  }
#endif  // CLIO_COMPRESSOR_ENABLE_DENSE_NN

  // Load NeuroPress NN model if configured (issue #693 Cycle 3). Consulted
  // first in EstCompressionStats()'s dynamic-selection path -- see there.
  if (!config_.neuropress_model_path_.empty()) {
    try {
      HLOG(kDebug, "Loading NeuroPress NN model from: {}",
           config_.neuropress_model_path_);
      neuropress_predictor_ =
          std::make_unique<ctp::compress::model::NeuroPressNNPredictor>();
      if (neuropress_predictor_->Load(config_.neuropress_model_path_)) {
        HLOG(kDebug, "NeuroPress NN model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load NeuroPress NN model from: {}",
             config_.neuropress_model_path_);
        neuropress_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading NeuroPress NN model: {}", e.what());
      neuropress_predictor_.reset();
    }
  }

  if (!qtable_predictor_ && !linreg_predictor_ && !neuropress_predictor_) {
    HLOG(kDebug,
         "No compression predictor configured, dynamic compression prediction "
         "disabled");
  }

  HLOG(kDebug,
       "CTE Compressor container created and initialized for pool: {} (ID: {})",
       pool_name_, pool_id_);

  // Spawn the periodic consumer-poll task (5s period). It iterates this
  // container's consumer list and dispatches PollNodeLoad to each node.
  client_.AsyncPollConsumers(clio::run::PoolQuery::Local(), 5000000);

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Reset predictors
    qtable_predictor_.reset();
    linreg_predictor_.reset();
    neuropress_predictor_.reset();
    // No distribution_classifier_ to reset

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
    nn_predictor_.reset();
#endif

    // Clear compression telemetry log if allocated
    // ShmPtr cleanup handled automatically

    HLOG(kDebug, "CTE Compressor container destroyed successfully");
  } catch (const std::exception& e) {
    HLOG(kError, "Exception during compressor destroy: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::PoolQuery Runtime::ScheduleTask(const clio::run::shared_ptr<clio::run::Task> &task) {
  // Compress placement: consult per-tag consumer tracking (when enabled)
  // so the compressed copy lands on the node that most recently read
  // the tag. Falls through to DirectHash(tag_id) when tracking is off
  // or the tag has no known consumers yet — keeps placement
  // deterministic per tag without the tracking overhead.
  if (task->method_ == Method::kCompress) {
    auto& compress_task = task.template Cast<CompressTask>();
    clio::run::u32 consumer_node = 0;
    if (PickConsumerForTag(compress_task->tag_id_, consumer_node)) {
      return clio::run::PoolQuery::Physical(consumer_node);
    }
    // No consumer info — hash on tag_id so all blobs of the same tag
    // converge on the same container regardless of which node submits.
    clio::run::u32 hash = static_cast<clio::run::u32>(
        std::hash<clio::cte::core::TagId>{}(compress_task->tag_id_));
    return clio::run::PoolQuery::DirectHash(hash);
  }
  // Other Dynamic methods (Decompress, periodic ticks) resolve Local.
  return clio::run::PoolQuery::Local();
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!core_client_) {
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  // Poll target states
  try {
    auto list_task = core_client_->AsyncListTargets();
    CLIO_CO_AWAIT(list_task);
    if (list_task->GetReturnCode() == 0) {
      std::lock_guard<std::mutex> lock(target_states_mutex_);
      for (auto &target_name : list_task->target_names_) {
        auto stat_task = core_client_->AsyncGetTargetInfo(target_name);
        CLIO_CO_AWAIT(stat_task);
        if (stat_task->GetReturnCode() == 0) {
          auto &state = target_states_[target_name];
          state.target_name_ = target_name;
          state.target_score_ = stat_task->target_score_;
          state.remaining_space_ = stat_task->remaining_space_;
          state.bytes_written_ = stat_task->bytes_written_;
        }
      }
    }
    // Serialize target_states_ to msgpack
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(target_states_.size());
    for (auto &[name, state] : target_states_) {
      pk.pack(name);
      pk.pack_map(4);
      pk.pack("score"); pk.pack(state.target_score_);
      pk.pack("remaining"); pk.pack(state.remaining_space_);
      pk.pack("written"); pk.pack(state.bytes_written_);
      pk.pack("name"); pk.pack(state.target_name_);
    }
    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  } catch (const std::exception &e) {
    HLOG(kError, "Compressor::Monitor failed: {}", e.what());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ==============================================================================
// Compression Statistics Estimation
// ==============================================================================

std::vector<CompressionStats> Runtime::EstCompressionStats(
    const void* chunk, clio::run::u64 chunk_size, const Context& context,
    bool* out_ranked_by_cost) {
  std::vector<CompressionStats> results;
  if (out_ranked_by_cost) *out_ranked_by_cost = false;

  // Determine data type from context
  // context.data_type_: 0 = char/uint8, 1 = float
  ctp::DataType data_type = (context.data_type_ == 1) ? ctp::DataType::FLOAT32
                                                       : ctp::DataType::UINT8;
  size_t type_size = ctp::DataStatisticsFactory::GetTypeSize(data_type);

  // Whole chunk, not a prefix. NeuroPress computes entropy/MAD/second-
  // derivative by grid-striding the ENTIRE buffer (stats_kernel.cu,
  // entropy_kernel.cu), and the shipped model was normalized against
  // whole-chunk statistics (x_means[4] ~ 1.4 MB). Sampling only the first
  // 64 KB fed the model prefix statistics paired with the FULL chunk_size
  // as the size feature -- a combination that never occurs in training, and
  // badly wrong for any chunk with a header, a zero-padded prologue, or
  // spatially varying structure. Since every candidate is scored from these
  // same three numbers, an error here shifts the whole ranking, not one
  // entry.
  size_t num_elements = static_cast<size_t>(chunk_size / type_size);
  if (num_elements == 0) {
    num_elements = 1;
  }

  // Calculate compression features. A chunk resolved from a CUDA-IPC device
  // buffer is not host-readable -- ComputeCompressionFeatures detects that
  // and computes entropy/MAD/second-derivative on-device, so the buffer
  // itself never has to be staged through host memory just to feed
  // NeuroPress. Falls through to the existing host path otherwise.
  double entropy = 0.0, mad = 0.0, second_derivative_mean = 0.0;
  ctp::ComputeCompressionFeatures(chunk, num_elements, data_type, &entropy,
                                   &mad, &second_derivative_mean);

  // Dynamic mode: NeuroPress (if configured) takes priority over the legacy
  // qtable/dense-NN heuristics below -- it ranks clio_ctp::compress::model's
  // candidate set restricted to its own trained 8-algorithm nvcomp GPU
  // action space (see NeuroPressCandidateStats), not just this function's
  // old 5-candidate hardcoded list. Every candidate it can return is
  // GPU-native, so a device-resident chunk_data never forces a host
  // round-trip downstream in Compress() regardless of where this chunk
  // happens to live.
  if (context.dynamic_compress_ != 1 && neuropress_predictor_ &&
      neuropress_predictor_->IsReady()) {
    bool data_type_float = (context.data_type_ == 1);
    auto neuropress_stats = NeuroPressCandidateStats(
        *neuropress_predictor_, chunk_size, entropy, mad,
        second_derivative_mean, data_type_float);
    // Apply the same PSNR filter the legacy per-candidate loop below uses,
    // since NeuroPressCandidateStats itself is PSNR-agnostic (issue #693).
    if (context.target_psnr_ > 0) {
      std::vector<CompressionStats> filtered;
      filtered.reserve(neuropress_stats.size());
      for (const auto& stat : neuropress_stats) {
        if (stat.psnr_db_ > 0 && stat.psnr_db_ < context.target_psnr_) continue;
        filtered.push_back(stat);
      }
      neuropress_stats = std::move(filtered);
    }
    if (!neuropress_stats.empty()) {
      if (out_ranked_by_cost) *out_ranked_by_cost = true;
      HLOG(kDebug,
           "NeuroPress dynamic selection: chunk_size={} entropy={} mad={} "
           "-> top pick wire_id={} ({} candidates ranked)",
           chunk_size, entropy, mad, neuropress_stats.front().compress_lib_,
           neuropress_stats.size());
      return neuropress_stats;
    }
    // Fall through to the legacy heuristics below if NeuroPress produced no
    // usable candidates.
  }

  // Determine candidate compression libraries and configs
  // Library IDs: BROTLI=0, BZIP2=1, Blosc2=2, FPZIP=3, LZ4=4, LZMA=5,
  //              SNAPPY=6, SZ3=7, ZFP=8, ZLIB=9, ZSTD=10
  // Config IDs: balanced=0, best=1, default=2, fast=3
  std::vector<std::pair<int, int>> candidate_lib_configs;
  if (context.dynamic_compress_ == 1) {
    // Static mode: use specified library with default config
    candidate_lib_configs.push_back({context.compress_lib_, 2});
  } else {
    // Dynamic mode: test common library/config combinations
    candidate_lib_configs = {
        {10, 0},  // ZSTD balanced
        {10, 3},  // ZSTD fast
        {4, 3},   // LZ4 fast
        {1, 1},   // BZIP2 best
        {9, 0},   // ZLIB balanced
    };
  }

  // Run predictions for each candidate library/config
  for (const auto& [lib_id, config_id] : candidate_lib_configs) {
    CompressionPrediction pred;

    // Use Q-table predictor if available (primary method)
    if (qtable_predictor_ && qtable_predictor_->IsReady()) {
      CompressionFeatures features;
      features.library_config_id = static_cast<double>(lib_id);
      features.chunk_size_bytes = static_cast<double>(chunk_size);
      features.shannon_entropy = entropy;
      features.mad = mad;
      features.second_derivative_mean = second_derivative_mean;
      // Set config encoding
      features.config_fast = (config_id == 3) ? 1 : 0;
      features.config_balanced = (config_id == 0) ? 1 : 0;
      features.config_best = (config_id == 1) ? 1 : 0;
      // Set data type encoding
      features.data_type_char = (context.data_type_ == 0) ? 1 : 0;
      features.data_type_float = (context.data_type_ == 1) ? 1 : 0;

      pred = qtable_predictor_->Predict(features);
    }
#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
    // Fallback to DNN if Q-table not available
    else if (nn_predictor_ && nn_predictor_->IsReady()) {
      CompressionFeatures features;
      features.library_config_id = static_cast<double>(lib_id);
      features.chunk_size_bytes = static_cast<double>(chunk_size);
      features.shannon_entropy = entropy;
      features.mad = mad;
      features.second_derivative_mean = second_derivative_mean;
      features.config_fast = (config_id == 3) ? 1 : 0;
      features.config_balanced = (config_id == 0) ? 1 : 0;
      features.config_best = (config_id == 1) ? 1 : 0;
      features.data_type_char = (context.data_type_ == 0) ? 1 : 0;
      features.data_type_float = (context.data_type_ == 1) ? 1 : 0;
      pred = nn_predictor_->Predict(features);
    }
#endif  // CLIO_COMPRESSOR_ENABLE_DENSE_NN
    else {
      // Heuristic fallback if no predictor available
      pred.compression_ratio = 2.0;
      pred.psnr_db = 0.0;
      pred.compression_time_ms = static_cast<double>(chunk_size) / 100000.0;
    }

    // Filter out compressions below PSNR threshold
    if (context.target_psnr_ > 0 && pred.psnr_db > 0 &&
        pred.psnr_db < context.target_psnr_) {
      continue;
    }

    // Add to results with library and preset
    results.emplace_back(lib_id, config_id, pred.compression_ratio,
                         pred.compression_time_ms, pred.compression_time_ms,
                         pred.psnr_db);
  }

  return results;
}

double Runtime::EstWorkflowCompressTime(clio::run::u64 chunk_size, double tier_bw,
                                        const CompressionStats& stats,
                                        const Context& context) {
  double compressed_size = chunk_size / stats.compression_ratio_;
  double transfer_time_ms = (compressed_size / tier_bw) * 1000.0;

  if (stats.psnr_db_ == 0.0) {
    // Lossless compression
    return stats.compress_time_ms_ + stats.decompress_time_ms_ +
           transfer_time_ms;
  } else {
    // Lossy compression - may need verification decompression
    double psnr_check_prob = static_cast<double>(context.psnr_chance_) / 100.0;
    return stats.compress_time_ms_ +
           (1.0 + psnr_check_prob) * stats.decompress_time_ms_ +
           transfer_time_ms;
  }
}

std::tuple<int, int, int, double, float> Runtime::BestCompressRatio(
    const void* chunk, clio::run::u64 chunk_size, int container_id,
    const std::vector<CompressionStats>& stats, const Context& context) {
  int best_tier = 0;
  int best_lib = 0;
  int best_preset = 2;  // Default: BALANCED
  double best_time = std::numeric_limits<double>::max();
  double best_ratio = 1.0;
  float best_tier_score = 0.0F;

  // Get target bandwidth from cached target states
  double tier_bw = 1e9;  // Default: 1 GB/s
  {
    std::lock_guard<std::mutex> lock(target_states_mutex_);
    if (!target_states_.empty()) {
      // Find target with highest score (best performance)
      float max_score = 0.0F;
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > max_score) {
          max_score = state.target_score_;
          best_tier_score = max_score;
          // Estimate bandwidth from normalized log score
          // score = log(bw+1) / log(1000+1), solve for bw
          tier_bw = std::pow(1001.0, max_score) - 1.0;
          tier_bw = std::max(tier_bw, 1e6);   // At least 1 MB/s
          tier_bw = std::min(tier_bw, 1e10);  // Cap at 10 GB/s
        }
      }
    }
  }

  for (const auto& stat : stats) {
    // Calculate workflow time for this compression
    double est_time =
        EstWorkflowCompressTime(chunk_size, tier_bw, stat, context);

    // Choose compression with best ratio that meets time constraints
    if (stat.compression_ratio_ > best_ratio) {
      best_ratio = stat.compression_ratio_;
      best_lib = stat.compress_lib_;
      best_preset = stat.compress_preset_;
      best_time = est_time;
      best_tier = 0;
    }
  }

  return std::make_tuple(best_tier, best_lib, best_preset, best_time,
                         best_tier_score);
}

std::tuple<int, int, int, double, float> Runtime::BestCompressTime(
    const void* chunk, clio::run::u64 chunk_size, int container_id,
    const std::vector<CompressionStats>& stats, const Context& context) {
  int best_tier = 0;
  int best_lib = 0;
  int best_preset = 2;  // Default: BALANCED
  double best_time = std::numeric_limits<double>::max();
  float best_tier_score = 0.0F;

  // Get target bandwidth from cached target states
  double tier_bw = 1e9;  // Default: 1 GB/s
  {
    std::lock_guard<std::mutex> lock(target_states_mutex_);
    if (!target_states_.empty()) {
      // Find target with highest score (best performance)
      float max_score = 0.0F;
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > max_score) {
          max_score = state.target_score_;
          best_tier_score = max_score;
          // Estimate bandwidth from normalized log score
          // score = log(bw+1) / log(1000+1), solve for bw
          tier_bw = std::pow(1001.0, max_score) - 1.0;
          tier_bw = std::max(tier_bw, 1e6);   // At least 1 MB/s
          tier_bw = std::min(tier_bw, 1e10);  // Cap at 10 GB/s
        }
      }
    }
  }

  // For each compression library and tier, calculate workflow time
  for (const auto& stat : stats) {
    double est_time =
        EstWorkflowCompressTime(chunk_size, tier_bw, stat, context);

    // Choose combination with best performance
    if (est_time < best_time) {
      best_time = est_time;
      best_lib = stat.compress_lib_;
      best_preset = stat.compress_preset_;
      best_tier = 0;
    }
  }

  return std::make_tuple(best_tier, best_lib, best_preset, best_time,
                         best_tier_score);
}

std::tuple<int, int, int, double, float> Runtime::BestCompressForNode(
    const Context& context, const void* chunk, clio::run::u64 chunk_size,
    int container_id, const std::vector<CompressionStats>& stats) {
  // Choose strategy based on context objective
  if (context.max_performance_) {
    // Objective: minimize time
    return BestCompressTime(chunk, chunk_size, container_id, stats, context);
  }
  // Objective: maximize compression ratio
  return BestCompressRatio(chunk, chunk_size, container_id, stats, context);
}

// ==============================================================================
// Task Execution Methods
// ==============================================================================

// Static atomic trace key counter for generating unique trace IDs
static std::atomic<clio::run::u64> g_trace_key_counter{1};

// Helper function to write trace log entry
static void WriteTraceLog(const std::string& trace_folder,
                          const std::string& log_name, clio::run::u32 container_id,
                          const std::string& entry) {
  if (trace_folder.empty()) return;

  try {
    std::string log_path =
        trace_folder + "/" + log_name + "." + std::to_string(container_id);
    std::ofstream log_file(log_path, std::ios::app);
    if (log_file.is_open()) {
      log_file << entry << std::endl;
      log_file.close();
    }
  } catch (const std::exception& e) {
    HLOG(kWarning, "Failed to write trace log: {}", e.what());
  }
}

void Runtime::RecordDecompFeatures(
    const std::string& blob_key,
    const ctp::compress::model::CompressionFeatures& features) {
  std::lock_guard<std::mutex> lock(decomp_features_mutex_);
  auto it = decomp_features_.find(blob_key);
  if (it != decomp_features_.end()) {
    // Overwrite: the newest compression of this blob is what a subsequent
    // read will actually decompress.
    it->second.features = features;
    it->second.seq = decomp_feature_seq_++;
    return;
  }
  if (decomp_features_.size() >= kMaxDecompFeatureRecords) {
    // FIFO-evict the oldest. These are training hints for blobs that may
    // never be read back; dropping the stalest one loses a learning
    // opportunity, never correctness.
    auto oldest = decomp_features_.begin();
    for (auto cur = decomp_features_.begin(); cur != decomp_features_.end();
         ++cur) {
      if (cur->second.seq < oldest->second.seq) oldest = cur;
    }
    decomp_features_.erase(oldest);
  }
  decomp_features_.emplace(
      blob_key, DecompFeatureRecord{features, decomp_feature_seq_++});
}

void Runtime::LearnDecompTime(const std::string& blob_key,
                              double measured_ms) {
  if (!config_.neuropress_online_learning_enabled_ || measured_ms <= 0.0) {
    return;
  }
  if (!neuropress_predictor_ || !neuropress_predictor_->IsReady()) return;

  std::vector<ctp::compress::model::CompressionFeatures> batch_features;
  std::vector<double> batch_times;
  {
    std::lock_guard<std::mutex> lock(decomp_features_mutex_);
    auto it = decomp_features_.find(blob_key);
    if (it == decomp_features_.end()) return;  // never compressed here
    pending_decomp_features_.push_back(it->second.features);
    pending_decomp_times_.push_back(measured_ms);
    // Consume it: one measured read trains once. Leaving it would retrain
    // the same (features, time) pair on every subsequent read of the blob.
    decomp_features_.erase(it);

    if (pending_decomp_features_.size() < kDecompBatchSize) return;
    batch_features.swap(pending_decomp_features_);
    batch_times.swap(pending_decomp_times_);
  }

  // One averaged update over the batch, matching upstream's
  // gpucompress_batched_decomp_sgd() -- the trust region scales to the mean
  // absolute error across chunks, not to whichever single blob was read last.
  bool trained =
      neuropress_predictor_->TrainDecompHead(batch_features, batch_times);
  HLOG(kDebug, "NeuroPress decomp-head SGD: batch={} trained={}",
       batch_features.size(), trained);
}

clio::run::TaskResume Runtime::DynamicSchedule(
    clio::run::shared_ptr<DynamicScheduleTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract task parameters (same as PutBlobTask)
    clio::run::u64 chunk_size = task->size_;
    // Convert ShmPtr to raw pointer via FullPtr
    auto blob_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    void* chunk_data = blob_fullptr.ptr_;
    Context& context = task->context_;

    // Initialize tracing if enabled
    auto start_time = std::chrono::high_resolution_clock::now();
    if (context.trace_) {
      context.trace_key_ = g_trace_key_counter.fetch_add(1);
      context.trace_node_ = static_cast<int>(CLIO_IPC->GetNodeId());
    }

    // Check if we have valid chunk data
    if (chunk_data == nullptr || chunk_size == 0) {
      HLOG(kWarning, "Invalid chunk data for dynamic scheduling");
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Get compression stats
    bool ranked_by_cost = false;
    auto stats =
        EstCompressionStats(chunk_data, chunk_size, context, &ranked_by_cost);

    if (stats.empty()) {
      // No valid compression available, disable compression
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // Log predicted compression stats if tracing enabled
    if (context.trace_ && !stats.empty()) {
      for (const auto& stat : stats) {
        std::ostringstream log_entry;
        log_entry << context.trace_key_ << "," << stat.compress_lib_ << ","
                  << stat.compression_ratio_ << "," << stat.compress_time_ms_
                  << "," << stat.decompress_time_ms_ << "," << stat.psnr_db_;
        WriteTraceLog(config_.trace_folder_path_, "predicted_stats.log",
                      pool_id_.major_, log_entry.str());
      }
    }

    // Choose best compression strategy. When NeuroPress ranked these, the
    // list is ALREADY best-first under its cost model (comp time + decomp
    // time + I/O); re-running BestCompressForNode would re-select on ratio
    // alone and throw that ordering away. Take element 0 instead, which is
    // what NeuroPress's own action selection does.
    int best_tier = 0, best_lib = 0, best_preset = 2;
    double best_time = 0.0;
    float tier_score = 0.0F;
    if (ranked_by_cost) {
      best_lib = stats.front().compress_lib_;
      best_preset = stats.front().compress_preset_;
      best_time = stats.front().compress_time_ms_;
    } else {
      std::tie(best_tier, best_lib, best_preset, best_time, tier_score) =
          BestCompressForNode(context, chunk_data, chunk_size, container_id_,
                              stats);
    }

    // Update context with selected compression library and preset
    context.compress_lib_ = best_lib;
    context.compress_preset_ = best_preset;
    task->tier_score_ = tier_score;

    // Log scheduling decision time if tracing enabled
    if (context.trace_) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration_ms =
          std::chrono::duration<double, std::milli>(end_time - start_time)
              .count();

      std::ostringstream log_entry;
      log_entry << context.trace_key_ << "," << duration_ms;
      WriteTraceLog(config_.trace_folder_path_, "sched_decision.log",
                    pool_id_.major_, log_entry.str());
    }

    // Cycle 4f: snapshot the intrinsic data features NeuroPress's own
    // prediction was based on, for a possible SGD update below. Computed here
    // (chunk_data is certainly still valid -- the same pointer
    // EstCompressionStats just read) rather than after the await, since
    // nothing downstream guarantees the input blob outlives Compress().
    bool neuropress_feat_valid = false;
    double neuropress_entropy = 0.0, neuropress_mad = 0.0,
           neuropress_second_deriv = 0.0;
    if (config_.neuropress_online_learning_enabled_ &&
        context.dynamic_compress_ != 1 && neuropress_predictor_ &&
        neuropress_predictor_->IsReady()) {
      ctp::DataType feat_type = (context.data_type_ == 1)
                                    ? ctp::DataType::FLOAT32
                                    : ctp::DataType::UINT8;
      size_t feat_type_size = ctp::DataStatisticsFactory::GetTypeSize(feat_type);
      // Whole chunk -- MUST match EstCompressionStats' scope above, or the
      // features SGD trains on are not the features inference predicted
      // from, and the model learns against a different input than it saw.
      size_t feat_num_elements = static_cast<size_t>(chunk_size / feat_type_size);
      if (feat_num_elements == 0) feat_num_elements = 1;
      ctp::ComputeCompressionFeatures(chunk_data, feat_num_elements, feat_type,
                                      &neuropress_entropy, &neuropress_mad,
                                      &neuropress_second_deriv);
      neuropress_feat_valid = true;
    }

    // Now call Compress to perform compression and PutBlob
    auto compress_task = client_.AsyncCompress(
        clio::run::PoolQuery::Local(), task->tag_id_, task->blob_name_.str(),
        task->offset_, task->size_, task->blob_data_, task->score_, context,
        task->flags_, task->core_pool_id_);
    CLIO_CO_AWAIT(compress_task);

    // Copy results back
    task->context_ = compress_task->context_;
    task->tier_score_ = compress_task->tier_score_;
    task->return_code_ = compress_task->return_code_;

    // Cycle 4f/4g: NeuroPress's own online-learning loop
    // (gpucompress_compress.cpp), ported faithfully. Both phases below share
    // one error_pct (weighted-cost MAPE between what was predicted for the
    // algorithm actually used and what really happened) and only differ in
    // which threshold gates them -- exactly mirroring
    // g_reinforce_mape_threshold vs g_exploration_threshold.
    if (config_.neuropress_online_learning_enabled_ && neuropress_feat_valid &&
        task->return_code_ == 0) {
      const CompressionStats* predicted = nullptr;
      for (const auto& stat : stats) {
        if (stat.compress_lib_ == best_lib &&
            stat.compress_preset_ == best_preset) {
          predicted = &stat;
          break;
        }
      }
      if (predicted) {
        // cost = w0*compress_time + w1*decompress_time +
        //        w2*chunk_size/(ratio*bandwidth) -- same formula and default
        // weights/bandwidth as NeuroPress's own g_rank_w0/w1/w2 (all 1.0) and
        // g_measured_bw_bytes_per_ms (5e6 = 5 GB/s).
        constexpr double kCostW0 = 1.0, kCostW1 = 1.0, kCostW2 = 1.0;
        constexpr double kCostBandwidthBytesPerMs = 5e6;
        auto cost = [&](double compress_ms, double decompress_ms,
                        double ratio) -> double {
          double ct = std::max(1.0, compress_ms);
          double dt = std::max(1.0, decompress_ms);
          double rc = std::min(100.0, ratio);
          return kCostW0 * ct + kCostW1 * dt +
                 ((rc > 0.0) ? kCostW2 * static_cast<double>(chunk_size) /
                                   (rc * kCostBandwidthBytesPerMs)
                             : 1e30);
        };
        // Decompress time is not measured at write time (only a later read
        // decompresses it) -- use the prediction for both sides, same as
        // NeuroPress's own primary_decomp_time_ms fallback, so this term
        // contributes ~0 error rather than penalizing an unmeasured value.
        double predicted_cost = cost(predicted->compress_time_ms_,
                                     predicted->decompress_time_ms_,
                                     predicted->compression_ratio_);
        double actual_cost = cost(context.actual_compress_time_ms_,
                                  predicted->decompress_time_ms_,
                                  context.actual_compression_ratio_);
        double error_pct = (actual_cost > 0.0)
            ? std::fabs(actual_cost - predicted_cost) / actual_cost
            : 0.0;

        // ---- Phase 1: "learn from PRIMARY result immediately" -- online
        // SGD on the real, just-measured outcome for the algorithm that was
        // ACTUALLY used, gated by g_reinforce_mape_threshold / default 30%.
        // In-memory only: Train() adjusts neuropress_predictor_'s live
        // weights for this process's lifetime and never calls Save() -- the
        // .nnwt file on disk stays untouched, exactly like every NeuroPress
        // runtime API (only gpucompress_load_nn/_reload_nn touch the file,
        // and both are read-only).
        std::string lib_name =
            ctp::CompressionFactory::NameForWireId(best_lib);
        int base_id = -1;
        for (const auto& entry : ctp::compress::model::KnownCompressors()) {
          if (lib_name == entry.name) {
            base_id = entry.base_id;
            break;
          }
        }
        if (base_id >= 0) {
          ctp::compress::model::DataFeatures data;
          data.chunk_size_bytes = static_cast<double>(chunk_size);
          data.shannon_entropy = neuropress_entropy;
          data.mad = neuropress_mad;
          data.second_derivative_mean = neuropress_second_deriv;
          data.data_type_char = (context.data_type_ == 1) ? 0.0 : 1.0;
          data.data_type_float = (context.data_type_ == 1) ? 1.0 : 0.0;

          ctp::compress::model::CandidateConfig candidate;
          candidate.base_id = base_id;
          candidate.preset_id = best_preset;
          candidate.library_name = lib_name;

          ctp::compress::model::CompressionFeatures chunk_features =
              ctp::compress::model::MakeCompressionFeatures(data, candidate);

          // Stash for the deferred decomp-head pass. Recorded on EVERY
          // compress, not just when the MAPE gate below trips: any blob may
          // be read back later, and that read is the only place a real
          // decompression time ever becomes available.
          RecordDecompFeatures(task->blob_name_.str(), chunk_features);

          if (error_pct >
              static_cast<double>(config_.neuropress_mape_threshold_)) {
            std::vector<ctp::compress::model::CompressionFeatures> features = {
                chunk_features};
            std::vector<ctp::compress::model::TrainingLabels> labels = {
                ctp::compress::model::TrainingLabels(
                    static_cast<float>(context.actual_compression_ratio_),
                    static_cast<float>(context.actual_psnr_db_),
                    static_cast<float>(context.actual_compress_time_ms_),
                    /*decompress_time=*/0.0f)};

            bool trained = neuropress_predictor_->Train(features, labels);
            HLOG(kDebug,
                 "NeuroPress SGD: lib={} preset={} error_pct={} "
                 "threshold={} trained={}",
                 lib_name, best_preset, error_pct,
                 config_.neuropress_mape_threshold_, trained);
          }
        }

        // ---- Phase 2 (Cycle 4g): "learn from EXPLORATION results
        // separately" -- when error crossed the HIGHER exploration
        // threshold (default 50%, g_exploration_threshold), actually
        // compress the SAME chunk with up to K alternative candidates (the
        // next-best predicted ones, skipping whichever was used for the
        // real, stored compress) purely to generate more real-outcome
        // training samples. Never stored or returned -- the primary's
        // already-persisted result stays authoritative. Simplified from the
        // original's parallel-CUDA-stream implementation to Clio's own
        // synchronous per-candidate Compress() call: same data flow and
        // training outcome, sequential rather than stream-parallel --
        // exploration is opt-in (off by default, matching
        // g_exploration_enabled's own default) and off the storage
        // critical path either way.
        if (config_.neuropress_exploration_enabled_ &&
            error_pct >
                static_cast<double>(config_.neuropress_exploration_threshold_)) {
          std::vector<const CompressionStats*> alternatives;
          for (const auto& stat : stats) {
            if (stat.compress_lib_ == best_lib &&
                stat.compress_preset_ == best_preset) {
              continue;
            }
            alternatives.push_back(&stat);
            if (static_cast<int>(alternatives.size()) >=
                config_.neuropress_exploration_k_) {
              break;
            }
          }

          std::vector<ctp::compress::model::CompressionFeatures>
              explore_features;
          std::vector<ctp::compress::model::TrainingLabels> explore_labels;
          double best_cost = actual_cost;  // seeded with the primary's own

          for (const auto* alt : alternatives) {
            std::string alt_name =
                ctp::CompressionFactory::NameForWireId(alt->compress_lib_);
            ctp::CompressionPreset alt_preset =
                ctp::CompressionPreset::BALANCED;
            if (alt->compress_preset_ == 1) {
              alt_preset = ctp::CompressionPreset::FAST;
            } else if (alt->compress_preset_ == 3) {
              alt_preset = ctp::CompressionPreset::BEST;
            }
            auto alt_compressor =
                ctp::CompressionFactory::GetPreset(alt_name, alt_preset);
            if (!alt_compressor) continue;

            // Same device-pointer safety net Runtime::Compress() uses: a
            // CPU-only alternative can't read a device pointer directly.
            std::vector<char> alt_device_staging;
            char* alt_input = ctp::CompressionFactory::StageInputIfNeeded(
                static_cast<char*>(chunk_data), chunk_size,
                alt->compress_lib_, alt_device_staging);

            size_t alt_worst_case = chunk_size + (chunk_size / 20) + 1024;
            std::vector<char> alt_output(alt_worst_case);
            size_t alt_compressed_size = alt_worst_case;

            auto alt_start = std::chrono::high_resolution_clock::now();
            bool alt_ok = alt_compressor->Compress(
                alt_output.data(), alt_compressed_size, alt_input,
                chunk_size);
            double alt_time_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() -
                                     alt_start)
                                     .count();
            if (!alt_ok || alt_compressed_size == 0) continue;

            double alt_ratio = static_cast<double>(chunk_size) /
                               static_cast<double>(alt_compressed_size);
            double alt_cost =
                cost(alt_time_ms, alt->decompress_time_ms_, alt_ratio);
            if (alt_cost < best_cost) best_cost = alt_cost;

            int alt_base_id = -1;
            for (const auto& entry :
                ctp::compress::model::KnownCompressors()) {
              if (alt_name == entry.name) {
                alt_base_id = entry.base_id;
                break;
              }
            }
            if (alt_base_id < 0) continue;

            ctp::compress::model::DataFeatures alt_data;
            alt_data.chunk_size_bytes = static_cast<double>(chunk_size);
            alt_data.shannon_entropy = neuropress_entropy;
            alt_data.mad = neuropress_mad;
            alt_data.second_derivative_mean = neuropress_second_deriv;
            alt_data.data_type_char = (context.data_type_ == 1) ? 0.0 : 1.0;
            alt_data.data_type_float = (context.data_type_ == 1) ? 1.0 : 0.0;

            ctp::compress::model::CandidateConfig alt_candidate;
            alt_candidate.base_id = alt_base_id;
            alt_candidate.preset_id = alt->compress_preset_;
            alt_candidate.library_name = alt_name;

            explore_features.push_back(
                ctp::compress::model::MakeCompressionFeatures(alt_data,
                                                               alt_candidate));
            explore_labels.emplace_back(static_cast<float>(alt_ratio), 0.0f,
                                        static_cast<float>(alt_time_ms),
                                        0.0f);
          }

          if (!explore_features.empty()) {
            bool explore_trained =
                neuropress_predictor_->Train(explore_features, explore_labels);
            // Regret: how much worse the primary's real cost was than the
            // best alternative found. 0 if the primary was already best.
            double regret = (best_cost > 0.0)
                ? (actual_cost - best_cost) / best_cost
                : 0.0;
            HLOG(kDebug,
                 "NeuroPress explore: k={} error_pct={} threshold={} "
                 "trained={} regret={}",
                 explore_features.size(), error_pct,
                 config_.neuropress_exploration_threshold_, explore_trained,
                 regret);
          }
        }
      }
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in DynamicSchedule: {}", e.what());
    task->return_code_ = 1;
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Compress(clio::run::shared_ptr<CompressTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract task parameters (same as PutBlobTask)
    clio::run::u64 input_size = task->size_;
    Context& context = task->context_;

    // Validate inputs
    if (task->blob_data_.IsNull() || input_size == 0) {
      task->return_code_ = 1;  // Invalid input
      CLIO_CO_RETURN;
    }

    // Initialize core client if needed (from compose next_pool_id or task param)
    if (!core_client_) {
      clio::run::PoolId core_id = !config_.next_pool_id_.IsNull()
          ? config_.next_pool_id_ : task->core_pool_id_;
      if (!core_id.IsNull()) {
        core_client_ = std::make_unique<clio::cte::core::Client>(core_id);
      }
    }
    // Neither this chimod's own compose config nor the caller supplied a
    // core pool to store into -- every use of core_client_ below would
    // dereference a null unique_ptr. Fail the task instead of crashing the
    // whole runtime process on what is really a caller/deployment
    // misconfiguration (e.g. a compressor pool created without
    // CompressorConfig.next_pool_id_, called via the AsyncPutBlob
    // convenience override instead of AsyncDynamicSchedule/AsyncCompress'
    // explicit core_pool_id parameter).
    if (!core_client_) {
      HLOG(kError,
           "Compress: no core pool available (compose next_pool_id_ unset "
           "and caller passed no explicit core_pool_id) -- cannot store "
           "the result");
      task->return_code_ = 7;  // No core pool available
      CLIO_CO_RETURN;
    }

    // Get tier score for output
    float tier_score = 0.0F;
    {
      std::lock_guard<std::mutex> lock(target_states_mutex_);
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > tier_score) {
          tier_score = state.target_score_;
        }
      }
    }
    task->tier_score_ = tier_score;

    // If no compression requested, just call PutBlob directly
    if (context.compress_lib_ <= 0) {
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put_task);
      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;
      CLIO_CO_RETURN;
    }

    // Map the wire ID (CompressionHeader.compress_lib_) to a library name via
    // the shared registry in CompressionFactory (single source of truth; out-of-
    // range falls back to "zstd"). Note the wire ID is a separate namespace from
    // GetLibraryId's ML scheme (base_id*10 + preset, e.g. nvcomp-lz4 = 132).
    std::string library_name =
        ctp::CompressionFactory::NameForWireId(context.compress_lib_);

    // Map preset integer to enum
    // compress_preset_ carries the byte-shuffle element size in its high
    // bits (see PackPreset) -- unpack before mapping to the preset enum, or
    // a shuffled candidate reads as a bogus preset.
    const uint32_t packed_preset =
        static_cast<uint32_t>(context.compress_preset_);
    const uint32_t preset_id = UnpackPreset(packed_preset);
    const uint32_t shuffle_elem = UnpackShuffle(packed_preset);

    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (preset_id == 1) {
      preset = ctp::CompressionPreset::FAST;
    } else if (preset_id == 3) {
      preset = ctp::CompressionPreset::BEST;
    }

    // Create compressor with specified preset
    auto compressor = ctp::CompressionFactory::GetPreset(library_name, preset);

    if (!compressor) {
      HLOG(kWarning, "Failed to create compressor for library: {}",
           library_name);
      task->return_code_ = 3;  // Compressor creation failed
      CLIO_CO_RETURN;
    }

    size_t header_size = sizeof(CompressionHeader);
    // Worst-case compressed size: original size + 5% overhead.
    size_t worst_case_size = input_size + (input_size / 20) + 1024;

    // Convert ShmPtr to raw pointer via FullPtr
    auto input_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    char* input_ptr = input_fullptr.ptr_;

    // GPU-native libraries (nvcomp/cusz/cuszp/ndzip) accept a device pointer
    // directly -- they stage their own H2D as needed. Everything else is a
    // plain host codec that would read a device pointer directly and crash,
    // so stage a D2H copy ourselves first. NeuroPress's own selection keeps
    // this a safety net rather than the normal route: EstCompressionStats
    // excludes CPU candidates for a device-resident buffer, so only the
    // legacy heuristic fallback (or an explicit/static compress_lib_) can
    // still land here with a device pointer + CPU library.
    std::vector<char> device_staging;
    input_ptr = ctp::CompressionFactory::StageInputIfNeeded(
        input_ptr, input_size, context.compress_lib_, device_staging);

    // Byte-shuffle preprocessing, when the ranked candidate asked for it.
    // Groups each element's Nth byte together so a codec sees runs of
    // similar magnitude -- NeuroPress's own shuffle dimension. Applied AFTER
    // any device staging above, since ByteShuffle is host code, and the
    // element size is recorded in the header so Decompress can invert it.
    std::vector<char> shuffle_staging;
    uint32_t applied_shuffle = 0;
    if (shuffle_elem != 0 && input_size >= shuffle_elem &&
        (input_size % shuffle_elem) == 0) {
      std::vector<char> host_copy;
      const char *shuffle_src = input_ptr;
      if (ctp::IsDevicePointer(input_ptr)) {
        // A GPU-native codec left the buffer on the device; shuffling needs
        // it host-readable. Stage once here rather than silently skipping.
        host_copy.resize(input_size);
        ctp::GpuApi::Memcpy(host_copy.data(), input_ptr, input_size);
        shuffle_src = host_copy.data();
      }
      shuffle_staging.resize(input_size);
      if (ctp::compress::preprocess::ByteShuffle(reinterpret_cast<const uint8_t *>(shuffle_src),
                           input_size, shuffle_elem,
                           reinterpret_cast<uint8_t *>(
                               shuffle_staging.data()))) {
        input_ptr = shuffle_staging.data();
        applied_shuffle = shuffle_elem;
      }
      // On failure input_ptr is untouched and applied_shuffle stays 0, so
      // the header records "not shuffled" and the read side does nothing.
    }

    // input_ptr is still device-resident only when StageInputIfNeeded left
    // it alone, i.e. the codec is GPU-native. Runtime::Compress always runs
    // co-located with core_client_ (see above), so PutBlob below never
    // stages this output either -- it flows straight to
    // MemBdevTransport::WriteBlocks, which is already GPU-aware. Keeping the
    // compressed OUTPUT on-device too avoids compressor->Compress()'s own
    // internal D2H (see nvcomp.h's ToDeviceInput/out_is_device) plus this
    // function's own host memcpy into the SHM buffer below -- down to the
    // one D2H copy the bdev write does regardless.
    bool output_on_device = ctp::IsDevicePointer(input_ptr);

    std::vector<char> compressed_buffer;
    char *device_output = nullptr;
    ctp::ipc::AllocatorId device_output_alloc_id;
    if (output_on_device) {
      device_output_alloc_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          header_size + worst_case_size, &device_output);
      if (device_output_alloc_id.IsNull()) {
        output_on_device = false;  // Fall back to the host buffer below.
      }
    }
    if (!output_on_device) {
      compressed_buffer.resize(worst_case_size);
    }
    // Compressed bytes land after the header's spot so the device path can
    // fill the header in-place afterward without a second allocation.
    char *compress_dst =
        output_on_device ? (device_output + header_size)
                          : compressed_buffer.data();

    // Time ONLY the compress call. NeuroPress brackets exactly this with
    // CUDA events (gpucompress_compress.cpp) and its offline labels were
    // measured the same way, so including the output allocation and the
    // D2H/H2D staging above would feed the model a systematically inflated
    // comp_time -- biasing both the log1p target for output 0 and the
    // error_pct that gates whether training fires at all.
    size_t compressed_size = worst_case_size;
    auto compress_start = std::chrono::high_resolution_clock::now();
    bool success = compressor->Compress(compress_dst, compressed_size,
                                        input_ptr, input_size);

    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time =
        std::chrono::duration<double, std::milli>(compress_end - compress_start)
            .count();

    // Check if compression succeeded and is beneficial (include header size
    // in the total stored size)
    size_t total_stored_size = compressed_size + header_size;

    if (success && total_stored_size < input_size) {
      // Update context with compression statistics
      context.actual_original_size_ = input_size;
      context.actual_compressed_size_ = total_stored_size;
      context.actual_compression_ratio_ =
          static_cast<double>(input_size) /
          static_cast<double>(total_stored_size);
      context.actual_compress_time_ms_ = compress_time;

      // Record the shuffle that was ACTUALLY applied, not the one requested:
      // if ByteShuffle declined (unsupported element size, or a size that is
      // not a whole number of elements) the bytes are unshuffled and the
      // read side must not try to invert it.
      CompressionHeader header(context.compress_lib_,
                               PackPreset(preset_id, applied_shuffle),
                               input_size);
      ctp::ipc::ShmPtr<> compressed_shm_ptr;
      ctp::ipc::FullPtr<char> compressed_shm;  // Only used off the device path.

      if (output_on_device) {
        // Header goes in the room compress_dst was offset past above --
        // this is the only host touch for the whole compressed buffer, and
        // it's 24 bytes, not the payload.
        ctp::GpuApi::Memcpy(device_output,
                            reinterpret_cast<const char *>(&header),
                            header_size);
        compressed_shm_ptr.alloc_id_ = device_output_alloc_id;
        compressed_shm_ptr.off_ = reinterpret_cast<clio::run::u64>(device_output);
      } else {
        compressed_buffer.resize(compressed_size);

        // Allocate shared memory for header + compressed data
        compressed_shm = CLIO_IPC->AllocateBuffer(total_stored_size);
        if (compressed_shm.IsNull()) {
          HLOG(kError, "Failed to allocate shared memory for compressed data");
          task->return_code_ = 4;  // Memory allocation failed
          CLIO_CO_RETURN;
        }
        std::memcpy(compressed_shm.ptr_, &header, header_size);
        std::memcpy(compressed_shm.ptr_ + header_size, compressed_buffer.data(),
                    compressed_size);
        compressed_shm_ptr = compressed_shm.shm_.template Cast<void>();
      }

      // Tell the runtime these bytes are no longer the caller's bytes, so it
      // can mark the blob authoritatively (issue #818). This is the ONLY place
      // that knows it for certain -- compress_lib_ is set on the not-beneficial
      // path below too, where the stored bytes are raw.
      context.transform_flags_ |= clio::cte::core::kBlobTransformed |
                                  clio::cte::core::kBlobTransformCompressed;

      // Call PutBlob with header + compressed data
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_,
          total_stored_size, compressed_shm_ptr, task->score_, context,
          task->flags_, clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put_task);

      // Free the compressed-data buffer
      if (output_on_device) {
        CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, device_output_alloc_id);
      } else {
        CLIO_IPC->FreeBuffer(compressed_shm);
      }

      // Log compression telemetry
      CompressionTelemetry telemetry(
          CteOp::kPutBlob, context.compress_lib_, input_size, total_stored_size,
          compress_time, 0.0, 0.0, std::chrono::steady_clock::now(),
          compression_logical_time_.fetch_add(1));
      LogCompressionTelemetry(telemetry);

      HLOG(kDebug,
           "Compression: {} bytes -> {} bytes (ratio: {:.2f}, time: {:.2f}ms)",
           input_size, total_stored_size,
           static_cast<double>(input_size) /
               static_cast<double>(total_stored_size),
           compress_time);

      task->context_ = context;
      task->return_code_ = put_task->return_code_;
    } else {
      // Compression failed or didn't reduce size - store original data
      HLOG(kDebug, "Compression not beneficial, storing original data");

      // Mark uncompressed BEFORE the put, not after. The bytes below are the
      // caller's original bytes, so the blob must not be recorded as carrying
      // the codec we merely attempted -- this used to be zeroed only after
      // PutBlob had already copied compress_lib_ onto the BlobInfo, which is
      // half of why compress_lib_ cannot be trusted as a transform signal
      // (issue #818). transform_flags_ is deliberately left unset here.
      context.compress_lib_ = 0;

      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put_task);

      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Compress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Decompress(clio::run::shared_ptr<DecompressTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Record the originating node (the consumer that issued this Decompress)
    // against this specific tag. pool_query_.ret_node_ was stamped by the
    // sender's IpcManager when the task was first resolved, so it carries
    // the original sender's node id even after a network hop. Per-tag
    // tracking lets ScheduleTask later route Compress for the same tag
    // toward this reader. No-op when tracking_enabled_=false.
    RegisterConsumer(task->tag_id_, task->pool_query_.GetReturnNode());

    // Extract task parameters (same as GetBlobTask). task->size_ is the
    // caller-declared LOGICAL (original, uncompressed) size -- compression
    // is only ever kept when it makes the stored blob strictly smaller (see
    // Compress()'s "total_stored_size < input_size" check), so it is a safe
    // UPPER bound, but it is not the actual number of bytes physically
    // stored. Using it directly as compressed_size below over-reads past the
    // true compressed stream into the destination buffer's uninitialized
    // tail, which corrupts the decompressor's input; LZ4 in particular then
    // returns a negative status that silently underflows the unsigned
    // output_size and gets reported as success with a garbage size.
    clio::run::u64 expected_size = task->size_;

    // Validate output buffer
    if (task->blob_data_.IsNull()) {
      task->return_code_ = 1;  // Invalid output buffer
      CLIO_CO_RETURN;
    }

    // Initialize core client if needed (from compose next_pool_id or task param)
    if (!core_client_) {
      clio::run::PoolId core_id = !config_.next_pool_id_.IsNull()
          ? config_.next_pool_id_ : task->core_pool_id_;
      if (!core_id.IsNull()) {
        core_client_ = std::make_unique<clio::cte::core::Client>(core_id);
      }
    }
    // See the matching guard in Compress(): without a resolved core pool,
    // the unconditional core_client_->AsyncGetBlob() below null-derefs.
    if (!core_client_) {
      HLOG(kError,
           "Decompress: no core pool available (compose next_pool_id_ unset "
           "and caller passed no explicit core_pool_id) -- cannot fetch the "
           "blob to decompress");
      task->return_code_ = 3;  // No core pool available
      CLIO_CO_RETURN;
    }

    // Ask core directly (bypassing this class's own GetBlobSize override,
    // which deliberately reports the LOGICAL size) for the blob's actual
    // PHYSICAL size -- the true number of bytes to read and hand to the
    // decompressor. Falls back to the logical-size upper bound if the query
    // fails, so a size-lookup problem degrades to the old (possibly
    // over-reading) behavior rather than aborting the read outright.
    if (core_client_) {
      auto size_task = core_client_->AsyncGetBlobSize(
          task->tag_id_, task->blob_name_.str(), clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(size_task);
      if (size_task->return_code_ == 0 && size_task->size_ > 0) {
        expected_size = size_task->size_;
      }
    }

    // Allocate temporary buffer to receive compressed data from GetBlob,
    // sized to the blob's actual physical size (see above).
    auto temp_buffer = CLIO_IPC->AllocateBuffer(expected_size);
    if (temp_buffer.IsNull()) {
      task->return_code_ = 2;  // Memory allocation failed
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> temp_buffer_ptr = temp_buffer.shm_.template Cast<void>();

    // Call GetBlob to retrieve the (potentially compressed) data
    auto get_task = core_client_->AsyncGetBlob(
        task->tag_id_, task->blob_name_.str(), task->offset_, expected_size,
        task->flags_, temp_buffer_ptr, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(get_task);

    if (get_task->return_code_ != 0) {
      CLIO_IPC->FreeBuffer(temp_buffer);
      task->return_code_ = 10 + get_task->return_code_;  // GetBlob failed
      CLIO_CO_RETURN;
    }

    // Check for compression header
    auto* header = reinterpret_cast<CompressionHeader*>(temp_buffer.ptr_);
    size_t header_size = sizeof(CompressionHeader);

    if (header->IsValid()) {
      // Data is compressed - decompress it
      int compress_lib = static_cast<int>(header->compress_lib_);
      // High bits carry the byte-shuffle element size (see PackPreset). A
      // blob written before shuffling existed has zeros there, so it decodes
      // as "not shuffled" and this stays backward compatible.
      const uint32_t packed_preset = header->compress_preset_;
      int compress_preset = static_cast<int>(UnpackPreset(packed_preset));
      const uint32_t stored_shuffle = UnpackShuffle(packed_preset);
      clio::run::u64 original_size = header->original_size_;

      // Map the wire ID to a library name via the shared registry (single
      // source of truth; out-of-range falls back to "zstd"). Wire ID is a
      // separate namespace from the factory's ML scheme (base_id*10+preset).
      std::string library_name =
          ctp::CompressionFactory::NameForWireId(compress_lib);

      // Map preset integer to enum
      ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
      if (compress_preset == 1) {
        preset = ctp::CompressionPreset::FAST;
      } else if (compress_preset == 3) {
        preset = ctp::CompressionPreset::BEST;
      }

      // Create decompressor
      auto decompressor =
          ctp::CompressionFactory::GetPreset(library_name, preset);
      if (!decompressor) {
        CLIO_IPC->FreeBuffer(temp_buffer);
        HLOG(kWarning, "Failed to create decompressor for library: {}",
             library_name);
        task->return_code_ = 3;  // Decompressor creation failed
        CLIO_CO_RETURN;
      }

      auto decompress_start = std::chrono::high_resolution_clock::now();

      // Get compressed data (after header)
      char* compressed_data = temp_buffer.ptr_ + header_size;
      size_t compressed_size = expected_size - header_size;

      // Decompress to output buffer
      auto output_fullptr =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      size_t decompressed_size = original_size;
      bool success =
          decompressor->Decompress(output_fullptr.ptr_, decompressed_size,
                                   compressed_data, compressed_size);

      auto decompress_end = std::chrono::high_resolution_clock::now();
      double decompress_time = std::chrono::duration<double, std::milli>(
                                   decompress_end - decompress_start)
                                   .count();

      CLIO_IPC->FreeBuffer(temp_buffer);

      // Invert the byte-shuffle the write side applied. Must happen before
      // the caller ever sees the buffer -- a shuffled image is not the
      // user's data, it just happens to be the same length.
      if (success && stored_shuffle != 0 && decompressed_size >= stored_shuffle &&
          (decompressed_size % stored_shuffle) == 0) {
        std::vector<char> unshuffled(decompressed_size);
        if (ctp::compress::preprocess::ByteUnshuffle(
                reinterpret_cast<const uint8_t *>(output_fullptr.ptr_),
                decompressed_size, stored_shuffle,
                reinterpret_cast<uint8_t *>(unshuffled.data()))) {
          ctp::DeviceAwareMemcpy(output_fullptr.ptr_, unshuffled.data(),
                                 decompressed_size);
        } else {
          HLOG(kError,
               "Decompress: byte-unshuffle failed (elem={} size={}) -- the "
               "returned buffer would be shuffled garbage, failing instead",
               stored_shuffle, decompressed_size);
          success = false;
        }
      }

      if (success) {
        task->output_size_ = decompressed_size;
        task->decompress_time_ms_ = decompress_time;

        // Deferred decomp-head learning: this is the ONLY point a real
        // decompression time exists. Join it back to the features the
        // original Compress predicted from and train that one head.
        LearnDecompTime(task->blob_name_.str(), decompress_time);

        // Log decompression telemetry
        CompressionTelemetry telemetry(
            CteOp::kGetBlob, compress_lib, decompressed_size, compressed_size,
            0.0, decompress_time, 0.0, std::chrono::steady_clock::now(),
            compression_logical_time_.fetch_add(1));
        LogCompressionTelemetry(telemetry);

        HLOG(kDebug, "Decompression: {} bytes -> {} bytes (time: {:.2f}ms)",
             compressed_size, decompressed_size, decompress_time);

        task->return_code_ = 0;  // Success
      } else {
        HLOG(kError, "Decompression failed");
        task->output_size_ = 0;
        task->decompress_time_ms_ = 0.0;
        task->return_code_ = 5;  // Decompression failed
      }
    } else {
      // No compression header - data is uncompressed
      // Copy directly to output buffer
      auto output_fullptr =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      std::memcpy(output_fullptr.ptr_, temp_buffer.ptr_, expected_size);
      CLIO_IPC->FreeBuffer(temp_buffer);

      task->output_size_ = expected_size;
      task->decompress_time_ms_ = 0.0;
      task->return_code_ = 0;  // Success (no decompression needed)

      HLOG(kDebug, "GetBlob (no compression): {} bytes", expected_size);
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Decompress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::LogCompressionTelemetry(const CompressionTelemetry& telemetry) {
  // Log to compression telemetry buffer if available
  if (!compression_telemetry_log_.IsNull()) {
    // TODO: Fix ShmPtr API for telemetry logging
    // compression_telemetry_log_->Push(telemetry);
  }

  // Log to trace file if tracing is enabled
  if (!config_.trace_folder_path_.empty()) {
    std::ostringstream log_entry;
    log_entry << telemetry.logical_time_ << "," << telemetry.compress_lib_
              << "," << telemetry.original_size_ << ","
              << telemetry.compressed_size_ << ","
              << telemetry.compress_time_ms_ << ","
              << telemetry.decompress_time_ms_ << "," << telemetry.psnr_db_;

    std::string log_name = (telemetry.op_ == CteOp::kPutBlob)
                               ? "compress_stats.log"
                               : "decompress_stats.log";
    WriteTraceLog(config_.trace_folder_path_, log_name, pool_id_.major_,
                  log_entry.str());
  }
}

clio::run::u64 Runtime::GetWorkRemaining() const {
  // Return 0 - compressor has no persistent work queue
  return 0;
}

// ==============================================================================
// Consumer Tracking
// ==============================================================================

void Runtime::RegisterConsumer(const clio::cte::core::TagId &tag_id,
                               clio::run::u32 node_id) {
  // Tracking knob: when off, no per-tag bookkeeping happens and
  // ScheduleTask falls through to DirectHash on the tag_id. Use this to
  // measure the overhead of the tracking mechanism itself, or for
  // workloads with no producer-consumer locality.
  if (!config_.tracking_enabled_) {
    return;
  }

  // Fast path: lookup under reader lock. The per-tag vector grows only
  // (entries are never removed), so a stale read at worst sends one
  // duplicate registration through the writer path — which the writer
  // re-check absorbs.
  {
    clio::run::ScopedCoRwReadLock read_lock(tag_consumers_lock_);
    auto it = tag_consumers_.find(tag_id);
    if (it != tag_consumers_.end()) {
      for (clio::run::u32 existing : it->second) {
        if (existing == node_id) {
          return;  // Already registered for this tag.
        }
      }
    }
  }

  // Writer path: insert/grow under exclusive lock. Re-check first (another
  // writer may have raced us); cap at kMaxConsumersPerTag.
  clio::run::ScopedCoRwWriteLock write_lock(tag_consumers_lock_);
  auto &slots = tag_consumers_[tag_id];
  for (clio::run::u32 existing : slots) {
    if (existing == node_id) {
      return;
    }
  }
  if (slots.size() >= kMaxConsumersPerTag) {
    HLOG(kDebug,
         "Compressor: consumer slot full for tag ({} entries), dropping node {}",
         slots.size(), node_id);
    return;
  }
  slots.push_back(node_id);
  HLOG(kDebug,
       "Compressor: registered consumer node {} for tag (slot {}/{})",
       node_id, slots.size(), kMaxConsumersPerTag);
}

bool Runtime::PickConsumerForTag(const clio::cte::core::TagId &tag_id,
                                 clio::run::u32 &node_id_out) {
  if (!config_.tracking_enabled_) {
    return false;
  }
  clio::run::ScopedCoRwReadLock read_lock(tag_consumers_lock_);
  auto it = tag_consumers_.find(tag_id);
  if (it == tag_consumers_.end() || it->second.empty()) {
    return false;
  }
  // Most-recent reader heuristic: the latest pushed entry is the most
  // recent reader of the tag. A future improvement is to fold in the
  // PollConsumers load samples and pick the least-loaded known reader,
  // but the most-recent heuristic is cheap and exploits temporal
  // locality (read-then-recompute patterns).
  node_id_out = it->second.back();
  return true;
}

// ==============================================================================
// Node Load Sampling
// ==============================================================================

clio::run::TaskResume Runtime::PollNodeLoad(clio::run::shared_ptr<PollNodeLoadTask> &task) {
  CLIO_TASK_BODY_BEGIN
  NodeLoadSample sample;
  auto* ipc_manager = CLIO_IPC;
  sample.node_id_ = ipc_manager ? static_cast<clio::run::u32>(ipc_manager->GetNodeId())
                                : 0;

  // CPU utilization since the last sample. Mutex protects prev_cpu_times_
  // because PollNodeLoad may run concurrently across workers.
  ctp::CpuTimes cur = ctp::SystemInfo::GetCpuTimes();
  {
    std::lock_guard<std::mutex> lk(cpu_times_mutex_);
    sample.cpu_usage_pct_ =
        ctp::SystemInfo::ComputeCpuUtilization(prev_cpu_times_, cur);
    prev_cpu_times_ = cur;
  }

  // AggregateOut worker load across all workers on this node.
  auto* orchestrator = CLIO_WORK_ORCHESTRATOR;
  if (orchestrator) {
    std::size_t num_workers = orchestrator->GetWorkerCount();
    sample.num_workers_ = static_cast<clio::run::u32>(num_workers);
    for (std::size_t i = 0; i < num_workers; ++i) {
      clio::run::Worker* worker = orchestrator->GetWorker(static_cast<clio::run::u32>(i));
      if (!worker) {
        continue;
      }
      clio::run::WorkerStats stats = worker->GetWorkerStats();
      sample.worker_load_us_ += stats.load_;
      sample.num_queued_tasks_ += stats.num_queued_tasks_;
      sample.num_blocked_tasks_ += stats.num_blocked_tasks_;
    }
  }

  task->sample_ = sample;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PollConsumers(clio::run::shared_ptr<PollConsumersTask> &task) {
  CLIO_TASK_BODY_BEGIN
  (void)task;
  // No-op when tracking is disabled.
  if (!config_.tracking_enabled_) {
    CLIO_CO_RETURN;
  }
  // Snapshot the union of consumers across all tags under the reader
  // lock so the periodic poll does not hold the lock while issuing
  // remote tasks. We dedupe to a single PollNodeLoad per node — readers
  // may appear in multiple tags' lists.
  std::vector<clio::run::u32> snapshot;
  {
    clio::run::ScopedCoRwReadLock read_lock(tag_consumers_lock_);
    std::unordered_set<clio::run::u32> dedup;
    for (const auto &kv : tag_consumers_) {
      for (clio::run::u32 node : kv.second) {
        if (dedup.insert(node).second) {
          snapshot.push_back(node);
        }
      }
    }
  }

  if (snapshot.empty()) {
    CLIO_CO_RETURN;
  }

  // Fan out one PollNodeLoad task per consumer node, then await each.
  std::vector<clio::run::Future<PollNodeLoadTask>> futures;
  futures.reserve(snapshot.size());
  for (clio::run::u32 node_id : snapshot) {
    futures.emplace_back(
        client_.AsyncPollNodeLoad(clio::run::PoolQuery::Physical(node_id)));
  }

  for (std::size_t i = 0; i < futures.size(); ++i) {
    auto& fut = futures[i];
    CLIO_CO_AWAIT(fut);
    if (fut->GetReturnCode() == 0) {
      const NodeLoadSample& s = fut->sample_;
      HLOG(kDebug,
           "Compressor: consumer node {} cpu={:.1f}% worker_load={:.1f}us "
           "queued={} blocked={} workers={}",
           snapshot[i], s.cpu_usage_pct_, s.worker_load_us_,
           s.num_queued_tasks_, s.num_blocked_tasks_, s.num_workers_);
    } else {
      HLOG(kDebug, "Compressor: PollNodeLoad to node {} failed (rc={})",
           snapshot[i], fut->GetReturnCode());
    }
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}


// ============================================================================
// Interposed core data verbs (issue #886): the compressor as a transparent
// interposer over the CTE core's task interface. Machinery (forwarding,
// batching, region iteration) comes from CoreInterposer + blob_batch.h —
// shared with the replication chimod; only the transform policy lives here.
// ============================================================================

bool Runtime::CompressIntoShm(clio::cte::core::Context &ctx, const char *src,
                              clio::run::u64 size,
                              ctp::ipc::FullPtr<char> *stored,
                              clio::run::u64 *stored_size) {
  std::string library_name =
      ctp::CompressionFactory::NameForWireId(ctx.compress_lib_);
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (ctx.compress_preset_ == 1) {
    preset = ctp::CompressionPreset::FAST;
  } else if (ctx.compress_preset_ == 3) {
    preset = ctp::CompressionPreset::BEST;
  }
  auto compressor = ctp::CompressionFactory::GetPreset(library_name, preset);
  if (!compressor) {
    return false;
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<char> compressed(size + (size / 20) + 1024);
  size_t compressed_size = compressed.size();
  if (!compressor->Compress(compressed.data(), compressed_size,
                            const_cast<char *>(src), size)) {
    return false;
  }
  size_t header_size = sizeof(CompressionHeader);
  size_t total = compressed_size + header_size;
  if (total >= size) {
    return false;  // not beneficial — caller stores raw
  }
  auto shm = CLIO_IPC->AllocateBuffer(total);
  if (shm.IsNull()) {
    return false;
  }
  CompressionHeader header(ctx.compress_lib_, ctx.compress_preset_, size);
  std::memcpy(shm.ptr_, &header, header_size);
  std::memcpy(shm.ptr_ + header_size, compressed.data(), compressed_size);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0)
                  .count();
  ctx.actual_original_size_ = size;
  ctx.actual_compressed_size_ = total;
  ctx.actual_compression_ratio_ =
      static_cast<double>(size) / static_cast<double>(total);
  ctx.actual_compress_time_ms_ = ms;
  // The one place that KNOWS the stored bytes are no longer the caller's
  // bytes (issue #818 authoritative transform bit).
  ctx.transform_flags_ |= clio::cte::core::kBlobTransformed |
                          clio::cte::core::kBlobTransformCompressed;
  *stored = shm;
  *stored_size = total;
  return true;
}

int Runtime::DecompressStored(const char *stored, clio::run::u64 stored_size,
                              char *dst, clio::run::u64 dst_cap,
                              clio::run::u64 *out_size) {
  if (stored_size < sizeof(CompressionHeader)) {
    return 6;
  }
  const auto *header = reinterpret_cast<const CompressionHeader *>(stored);
  if (!header->IsValid() || header->original_size_ > dst_cap) {
    return 6;
  }
  std::string library_name = ctp::CompressionFactory::NameForWireId(
      static_cast<int>(header->compress_lib_));
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (header->compress_preset_ == 1) {
    preset = ctp::CompressionPreset::FAST;
  } else if (header->compress_preset_ == 3) {
    preset = ctp::CompressionPreset::BEST;
  }
  auto decompressor = ctp::CompressionFactory::GetPreset(library_name, preset);
  if (!decompressor) {
    return 3;
  }
  size_t decompressed = header->original_size_;
  if (!decompressor->Decompress(dst, decompressed,
                                const_cast<char *>(stored) +
                                    sizeof(CompressionHeader),
                                stored_size - sizeof(CompressionHeader))) {
    return 5;
  }
  *out_size = decompressed;
  return 0;
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::cte::core::Context &ctx = task->context_;
    // Compression is defined for WHOLE-BLOB writes only: a partial or
    // vectored write cannot patch a compressed stream, so those (and
    // replica-addressed or emulated puts) forward with the codec request
    // cleared — raw bytes, never a recorded codec (issue #818 rule).
    const bool whole_blob = task->segments_.empty() && task->offset_ == 0;
    if (ctx.replica_ != 0 || ctx.compress_lib_ <= 0 || ctx.emulate_ ||
        !whole_blob) {
      if (ctx.compress_lib_ > 0 && !whole_blob) {
        ctx.compress_lib_ = 0;
      }
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    auto src_full =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    ctp::ipc::FullPtr<char> stored;
    clio::run::u64 stored_size = 0;
    if (src_full.ptr_ != nullptr &&
        CompressIntoShm(ctx, src_full.ptr_, task->size_, &stored,
                        &stored_size)) {
      if (!core_client_) {
        core_client_ =
            std::make_unique<clio::cte::core::Client>(CorePoolId());
      }
      auto put = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), 0, stored_size,
          stored.shm_.template Cast<void>(), task->score_, ctx, task->flags_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put);
      CLIO_IPC->FreeBuffer(stored);
      task->context_ = put->context_;
      task->return_code_ = put->GetReturnCode();
    } else {
      // Failed or not beneficial: store the caller's raw bytes and make
      // sure the blob is never recorded as carrying the attempted codec.
      ctx.compress_lib_ = 0;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Serve the read as-is first: the untransformed case (and every replica-
  // addressed read) costs nothing extra, and the core reports the blob's
  // authoritative transform state OUT through the context either way.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                         task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() != 0 || task->context_.replica_ != 0 ||
      !(task->context_.transform_flags_ &
        clio::cte::core::kBlobTransformCompressed)) {
    CLIO_CO_RETURN;
  }
  {
    // The forwarded read handed back CODEC bytes. Fetch the whole stored
    // blob, decompress ONCE, then slice every requested region out of the
    // original — which is what makes partial and VECTORED reads of
    // compressed blobs work through this interposer at all.
    if (!core_client_) {
      core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
    }
    clio::run::u64 stored_size = 0;
    {
      auto sz = core_client_->AsyncGetBlobSize(task->tag_id_,
                                               task->blob_name_.str());
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
        task->return_code_ = 10 + sz->GetReturnCode();
        CLIO_CO_RETURN;
      }
      stored_size = sz->size_;
    }
    auto stored = CLIO_IPC->AllocateBuffer(stored_size);
    if (stored.IsNull()) {
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    {
      auto get = core_client_->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0, stored_size,
          /*flags=*/0, stored.shm_.template Cast<void>(),
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() != 0) {
        CLIO_IPC->FreeBuffer(stored);
        task->return_code_ = 10 + get->GetReturnCode();
        CLIO_CO_RETURN;
      }
    }
    const auto *header =
        reinterpret_cast<const CompressionHeader *>(stored.ptr_);
    clio::run::u64 original_size =
        stored_size >= sizeof(CompressionHeader) ? header->original_size_ : 0;
    auto scratch = CLIO_IPC->AllocateBuffer(original_size);
    if (scratch.IsNull()) {
      CLIO_IPC->FreeBuffer(stored);
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    clio::run::u64 out_size = 0;
    int rc = DecompressStored(stored.ptr_, stored_size, scratch.ptr_,
                              original_size, &out_size);
    CLIO_IPC->FreeBuffer(stored);
    if (rc != 0) {
      CLIO_IPC->FreeBuffer(scratch);
      task->return_code_ = rc;
      CLIO_CO_RETURN;
    }
    // Copy each requested region out of the original bytes (shared
    // scalar-vs-vectored iteration, blob_batch.h). Regions beyond the
    // original size keep the core's short-read semantics (left untouched).
    bool region_ok = true;
    clio::cte::core::ForEachBlobRegion(
        *task, [&](const clio::cte::core::BlobRegion &r) {
          if (r.blob_off_ >= out_size) {
            return true;  // wholly past EOF: short read
          }
          clio::run::u64 n = r.size_;
          if (r.blob_off_ + n > out_size) {
            n = out_size - r.blob_off_;
          }
          auto dst =
              CLIO_IPC->ToFullPtr<char>(r.data_.template Cast<char>());
          if (dst.ptr_ == nullptr) {
            region_ok = false;
            return false;
          }
          std::memcpy(dst.ptr_, scratch.ptr_ + r.blob_off_, n);
          return true;
        });
    CLIO_IPC->FreeBuffer(scratch);
    // The caller now holds ORIGINAL bytes: clear the transform report so
    // nothing downstream tries to undo the codec again.
    task->context_.transform_flags_ &=
        ~(clio::cte::core::kBlobTransformed |
          clio::cte::core::kBlobTransformCompressed);
    task->return_code_ = region_ok ? 0 : 3;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                         task.template Cast<clio::run::Task>()));
  // A transformed blob's stored size is header+codec bytes; size-then-read
  // callers need the LOGICAL size. Probe the header with a tiny ranged get —
  // its OUT context carries the authoritative transform bit, so a raw blob
  // that merely looks like a header can never be misreported.
  if (task->GetReturnCode() == 0 && task->replica_ == 0 &&
      task->size_ >= sizeof(CompressionHeader)) {
    if (!core_client_) {
      core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
    }
    auto hdr_buf = CLIO_IPC->AllocateBuffer(sizeof(CompressionHeader));
    if (!hdr_buf.IsNull()) {
      auto get = core_client_->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0, sizeof(CompressionHeader),
          /*flags=*/0, hdr_buf.shm_.template Cast<void>(),
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() == 0 &&
          (get->context_.transform_flags_ &
           clio::cte::core::kBlobTransformCompressed)) {
        const auto *header =
            reinterpret_cast<const CompressionHeader *>(hdr_buf.ptr_);
        if (header->IsValid()) {
          task->size_ = header->original_size_;
        }
      }
      CLIO_IPC->FreeBuffer(hdr_buf);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batches carry a batch-wide Context (issue #886 follow-up) and get
  // SCALAR-EQUIVALENT semantics. No codec requested (or replica-addressed /
  // emulated): forward the batch intact — the chain below executes every
  // record with this context, records stay raw, amortization preserved.
  if (task->context_.replica_ != 0 || task->context_.compress_lib_ <= 0 ||
      task->context_.emulate_) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  // Codec requested: records transform INDIVIDUALLY (some compress, some
  // stay raw — not-beneficial or offset writes), so one shared batch cannot
  // describe the results. Decompose through OUR scalar handler, which
  // applies the exact scalar rules per record.
  task->num_ok_ = 0;
  task->first_rc_ = 0;
  {
    auto *ipc_manager = CLIO_CPU_IPC;
    clio::cte::core::MultiPutBatchView batch;
    if (!clio::cte::core::MultiPutBatchView::Attach(*task, &batch)) {
      task->SetReturnCode(batch.descs_.empty() ? 0 : 1);
      CLIO_CO_RETURN;
    }
    for (size_t bi = 0; bi < batch.size(); ++bi) {
      const auto &d = batch.descs_[bi];
      if (!batch.RecordValid(bi)) {
        if (task->first_rc_ == 0) task->first_rc_ = 2;
        continue;
      }
      auto sub = ipc_manager->NewTask<clio::cte::core::PutBlobTask>(
          clio::run::CreateTaskId(), task->pool_id_,
          clio::run::PoolQuery::Local(), d.tag_id_, d.blob_name_, d.offset_,
          d.size_, batch.RecordSlice(bi), /*score=*/-1.0f, task->context_,
          /*flags=*/0);
      sub.get()->BeginRunContext();
      CLIO_CO_AWAIT(PutBlob(sub));
      int rc = sub->GetReturnCode();
      if (rc == 0) {
        task->num_ok_++;
      } else if (task->first_rc_ == 0) {
        task->first_rc_ = rc;
      }
    }
  }
  task->SetReturnCode(task->first_rc_ == 0 ? 0 : task->first_rc_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::compressor

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::cte::compressor::Runtime)
