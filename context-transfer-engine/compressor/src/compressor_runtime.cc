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
#include <wrp_cte/compressor/compressor_runtime.h>

#include <hermes_shm/serialize/msgpack_wrapper.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "chimaera/worker.h"
#include "hermes_shm/compress/compress_factory.h"
#include "hermes_shm/compress/data_stats.h"
#include "hermes_shm/util/logging.h"

namespace wrp_cte::compressor {

// Bring chi namespace items into scope for CHI_CUR_WORKER macro
using chi::chi_cur_worker_key_;
using chi::Worker;

/**
 * Compression header prepended to compressed data for self-describing format.
 * This allows decompression without external metadata.
 */
struct CompressionHeader {
  static constexpr uint32_t kMagic = 0x43544543;  // "CTEC" in ASCII
  uint32_t magic_;            // Magic number to identify compressed data
  uint32_t compress_lib_;     // Compression library ID
  uint32_t compress_preset_;  // Compression preset
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

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task,
                                chi::RunContext& ctx) {
  // Note: This is a simplified Create task since we don't have a CreateTask
  // defined In the actual implementation, you would extract config from task
  // parameters

  // Initialize atomic counters
  compression_logical_time_ = 0;

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

#ifdef WRP_COMPRESSOR_ENABLE_DENSE_NN
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
#endif  // WRP_COMPRESSOR_ENABLE_DENSE_NN

  if (!qtable_predictor_ && !linreg_predictor_) {
    HLOG(kDebug,
         "No compression predictor configured, dynamic compression prediction "
         "disabled");
  }

  HLOG(kDebug,
       "CTE Compressor container created and initialized for pool: {} (ID: {})",
       pool_name_, pool_id_);

  co_return;
}

chi::TaskResume Runtime::Destroy(hipc::FullPtr<DestroyTask> task,
                                 chi::RunContext& ctx) {
  try {
    // Reset predictors
    qtable_predictor_.reset();
    linreg_predictor_.reset();
    // No distribution_classifier_ to reset

#ifdef WRP_COMPRESSOR_ENABLE_DENSE_NN
    nn_predictor_.reset();
#endif

    // Clear compression telemetry log if allocated
    // ShmPtr cleanup handled automatically

    HLOG(kDebug, "CTE Compressor container destroyed successfully");
  } catch (const std::exception& e) {
    HLOG(kError, "Exception during compressor destroy: {}", e.what());
  }
  co_return;
}

chi::PoolQuery Runtime::ScheduleTask(const hipc::FullPtr<chi::Task> &task) {
  // All compressor Dynamic methods resolve to Local
  return chi::PoolQuery::Local();
}

chi::TaskResume Runtime::Monitor(hipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &ctx) {
  if (!core_client_) {
    task->SetReturnCode(0);
    co_return;
  }
  // Poll target states
  try {
    auto list_task = core_client_->AsyncListTargets();
    list_task.Wait();
    if (list_task->GetReturnCode() == 0) {
      std::lock_guard<std::mutex> lock(target_states_mutex_);
      for (auto &target_name : list_task->target_names_) {
        auto stat_task = core_client_->AsyncGetTargetInfo(target_name);
        stat_task.Wait();
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
  co_return;
}

// ==============================================================================
// Compression Statistics Estimation
// ==============================================================================

std::vector<CompressionStats> Runtime::EstCompressionStats(
    const void* chunk, chi::u64 chunk_size, const Context& context) {
  std::vector<CompressionStats> results;

  // Determine data type from context
  // context.data_type_: 0 = char/uint8, 1 = float
  hshm::DataType data_type = (context.data_type_ == 1) ? hshm::DataType::FLOAT32
                                                       : hshm::DataType::UINT8;
  size_t type_size = hshm::DataStatisticsFactory::GetTypeSize(data_type);

  // Calculate number of elements (sample up to 64KB for efficiency)
  chi::u64 sample_bytes = std::min(chunk_size, static_cast<chi::u64>(65536));
  size_t num_elements = sample_bytes / type_size;
  if (num_elements == 0) {
    num_elements = 1;
  }

  // Calculate compression features using DataStatisticsFactory
  double entropy = hshm::DataStatisticsFactory::CalculateShannonEntropy(
      chunk, num_elements, data_type);
  double mad =
      hshm::DataStatisticsFactory::CalculateMAD(chunk, num_elements, data_type);
  double second_derivative_mean =
      hshm::DataStatisticsFactory::CalculateSecondDerivative(
          chunk, num_elements, data_type);

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
#ifdef WRP_COMPRESSOR_ENABLE_DENSE_NN
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
#endif  // WRP_COMPRESSOR_ENABLE_DENSE_NN
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

double Runtime::EstWorkflowCompressTime(chi::u64 chunk_size, double tier_bw,
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
    const void* chunk, chi::u64 chunk_size, int container_id,
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
    const void* chunk, chi::u64 chunk_size, int container_id,
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
    const Context& context, const void* chunk, chi::u64 chunk_size,
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
static std::atomic<chi::u64> g_trace_key_counter{1};

// Helper function to write trace log entry
static void WriteTraceLog(const std::string& trace_folder,
                          const std::string& log_name, chi::u32 container_id,
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

chi::TaskResume Runtime::DynamicSchedule(
    hipc::FullPtr<DynamicScheduleTask> task, chi::RunContext& ctx) {
  try {
    // Extract task parameters (same as PutBlobTask)
    chi::u64 chunk_size = task->size_;
    // Convert ShmPtr to raw pointer via FullPtr
    auto blob_fullptr =
        CHI_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    void* chunk_data = blob_fullptr.ptr_;
    Context& context = task->context_;

    // Initialize tracing if enabled
    auto start_time = std::chrono::high_resolution_clock::now();
    if (context.trace_) {
      context.trace_key_ = g_trace_key_counter.fetch_add(1);
      context.trace_node_ = static_cast<int>(CHI_IPC->GetNodeId());
    }

    // Check if we have valid chunk data
    if (chunk_data == nullptr || chunk_size == 0) {
      HLOG(kWarning, "Invalid chunk data for dynamic scheduling");
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 1;
      co_return;
    }

    // Get compression stats
    auto stats = EstCompressionStats(chunk_data, chunk_size, context);

    if (stats.empty()) {
      // No valid compression available, disable compression
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 0;
      co_return;
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

    // Choose best compression strategy
    auto [best_tier, best_lib, best_preset, best_time, tier_score] =
        BestCompressForNode(context, chunk_data, chunk_size, container_id_,
                            stats);

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

    // Now call Compress to perform compression and PutBlob
    auto compress_task = client_.AsyncCompress(
        chi::PoolQuery::Local(), task->tag_id_, task->blob_name_.str(),
        task->offset_, task->size_, task->blob_data_, task->score_, context,
        task->flags_, task->core_pool_id_);
    compress_task.Wait();

    // Copy results back
    task->context_ = compress_task->context_;
    task->tier_score_ = compress_task->tier_score_;
    task->return_code_ = compress_task->return_code_;

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in DynamicSchedule: {}", e.what());
    task->return_code_ = 1;
  }

  co_return;
}

chi::TaskResume Runtime::Compress(hipc::FullPtr<CompressTask> task,
                                  chi::RunContext& ctx) {
  try {
    // Extract task parameters (same as PutBlobTask)
    chi::u64 input_size = task->size_;
    Context& context = task->context_;

    // Validate inputs
    if (task->blob_data_.IsNull() || input_size == 0) {
      task->return_code_ = 1;  // Invalid input
      co_return;
    }

    // Initialize core client if needed
    if (!core_client_) {
      core_client_ =
          std::make_unique<wrp_cte::core::Client>(task->core_pool_id_);
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
          chi::PoolQuery::Local());
      put_task.Wait();
      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;
      co_return;
    }

    // Map compress_lib_ ID to library name
    const char* lib_names[] = {"brotli", "bzip2", "blosc2", "fpzip",
                               "lz4",    "lzma",  "snappy", "sz3",
                               "zfp",    "zlib",  "zstd"};
    std::string library_name =
        (context.compress_lib_ >= 0 && context.compress_lib_ <= 10)
            ? lib_names[context.compress_lib_]
            : "zstd";

    // Map preset integer to enum
    hshm::CompressionPreset preset = hshm::CompressionPreset::BALANCED;
    if (context.compress_preset_ == 1) {
      preset = hshm::CompressionPreset::FAST;
    } else if (context.compress_preset_ == 3) {
      preset = hshm::CompressionPreset::BEST;
    }

    // Create compressor with specified preset
    auto compressor = hshm::CompressionFactory::GetPreset(library_name, preset);

    if (!compressor) {
      HLOG(kWarning, "Failed to create compressor for library: {}",
           library_name);
      task->return_code_ = 3;  // Compressor creation failed
      co_return;
    }

    auto compress_start = std::chrono::high_resolution_clock::now();

    // Allocate buffer for compressed data (worst case: original size + 5%
    // overhead)
    std::vector<char> compressed_buffer(input_size + (input_size / 20) + 1024);

    // Compress the data
    size_t compressed_size = compressed_buffer.size();
    // Convert ShmPtr to raw pointer via FullPtr
    auto input_fullptr =
        CHI_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    char* input_ptr = input_fullptr.ptr_;
    bool success = compressor->Compress(compressed_buffer.data(),
                                        compressed_size, input_ptr, input_size);

    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time =
        std::chrono::duration<double, std::milli>(compress_end - compress_start)
            .count();

    // Check if compression succeeded and is beneficial
    // Include header size in the total stored size
    size_t header_size = sizeof(CompressionHeader);
    size_t total_stored_size = compressed_size + header_size;

    if (success && total_stored_size < input_size) {
      // Compression succeeded and reduced size (including header overhead)
      compressed_buffer.resize(compressed_size);

      // Update context with compression statistics
      context.actual_original_size_ = input_size;
      context.actual_compressed_size_ = total_stored_size;
      context.actual_compression_ratio_ =
          static_cast<double>(input_size) /
          static_cast<double>(total_stored_size);
      context.actual_compress_time_ms_ = compress_time;

      // Allocate shared memory for header + compressed data
      auto compressed_shm = CHI_IPC->AllocateBuffer(total_stored_size);
      if (compressed_shm.IsNull()) {
        HLOG(kError, "Failed to allocate shared memory for compressed data");
        task->return_code_ = 4;  // Memory allocation failed
        co_return;
      }

      // Write compression header
      CompressionHeader header(context.compress_lib_, context.compress_preset_,
                               input_size);
      std::memcpy(compressed_shm.ptr_, &header, header_size);

      // Write compressed data after header
      std::memcpy(compressed_shm.ptr_ + header_size, compressed_buffer.data(),
                  compressed_size);

      // Call PutBlob with header + compressed data
      hipc::ShmPtr<> compressed_shm_ptr =
          compressed_shm.shm_.template Cast<void>();
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_,
          total_stored_size, compressed_shm_ptr, task->score_, context,
          task->flags_, chi::PoolQuery::Local());
      put_task.Wait();

      // Free compressed data buffer
      CHI_IPC->FreeBuffer(compressed_shm);

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

      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          chi::PoolQuery::Local());
      put_task.Wait();

      context.compress_lib_ = 0;  // Mark as uncompressed
      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Compress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  co_return;
}

chi::TaskResume Runtime::Decompress(hipc::FullPtr<DecompressTask> task,
                                    chi::RunContext& ctx) {
  try {
    // Extract task parameters (same as GetBlobTask)
    chi::u64 expected_size = task->size_;

    // Validate output buffer
    if (task->blob_data_.IsNull()) {
      task->return_code_ = 1;  // Invalid output buffer
      co_return;
    }

    // Initialize core client if needed
    if (!core_client_) {
      core_client_ =
          std::make_unique<wrp_cte::core::Client>(task->core_pool_id_);
    }

    // Allocate temporary buffer to receive compressed data from GetBlob
    // We don't know the compressed size, so allocate expected_size as upper
    // bound
    auto temp_buffer = CHI_IPC->AllocateBuffer(expected_size);
    if (temp_buffer.IsNull()) {
      task->return_code_ = 2;  // Memory allocation failed
      co_return;
    }
    hipc::ShmPtr<> temp_buffer_ptr = temp_buffer.shm_.template Cast<void>();

    // Call GetBlob to retrieve the (potentially compressed) data
    auto get_task = core_client_->AsyncGetBlob(
        task->tag_id_, task->blob_name_.str(), task->offset_, expected_size,
        task->flags_, temp_buffer_ptr, chi::PoolQuery::Local());
    get_task.Wait();

    if (get_task->return_code_ != 0) {
      CHI_IPC->FreeBuffer(temp_buffer);
      task->return_code_ = 10 + get_task->return_code_;  // GetBlob failed
      co_return;
    }

    // Check for compression header
    auto* header = reinterpret_cast<CompressionHeader*>(temp_buffer.ptr_);
    size_t header_size = sizeof(CompressionHeader);

    if (header->IsValid()) {
      // Data is compressed - decompress it
      int compress_lib = static_cast<int>(header->compress_lib_);
      int compress_preset = static_cast<int>(header->compress_preset_);
      chi::u64 original_size = header->original_size_;

      // Map compress_lib ID to library name
      const char* lib_names[] = {"brotli", "bzip2", "blosc2", "fpzip",
                                 "lz4",    "lzma",  "snappy", "sz3",
                                 "zfp",    "zlib",  "zstd"};
      std::string library_name = (compress_lib >= 0 && compress_lib <= 10)
                                     ? lib_names[compress_lib]
                                     : "zstd";

      // Map preset integer to enum
      hshm::CompressionPreset preset = hshm::CompressionPreset::BALANCED;
      if (compress_preset == 1) {
        preset = hshm::CompressionPreset::FAST;
      } else if (compress_preset == 3) {
        preset = hshm::CompressionPreset::BEST;
      }

      // Create decompressor
      auto decompressor =
          hshm::CompressionFactory::GetPreset(library_name, preset);
      if (!decompressor) {
        CHI_IPC->FreeBuffer(temp_buffer);
        HLOG(kWarning, "Failed to create decompressor for library: {}",
             library_name);
        task->return_code_ = 3;  // Decompressor creation failed
        co_return;
      }

      auto decompress_start = std::chrono::high_resolution_clock::now();

      // Get compressed data (after header)
      char* compressed_data = temp_buffer.ptr_ + header_size;
      size_t compressed_size = expected_size - header_size;

      // Decompress to output buffer
      auto output_fullptr =
          CHI_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      size_t decompressed_size = original_size;
      bool success =
          decompressor->Decompress(output_fullptr.ptr_, decompressed_size,
                                   compressed_data, compressed_size);

      auto decompress_end = std::chrono::high_resolution_clock::now();
      double decompress_time = std::chrono::duration<double, std::milli>(
                                   decompress_end - decompress_start)
                                   .count();

      CHI_IPC->FreeBuffer(temp_buffer);

      if (success) {
        task->output_size_ = decompressed_size;
        task->decompress_time_ms_ = decompress_time;

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
          CHI_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      std::memcpy(output_fullptr.ptr_, temp_buffer.ptr_, expected_size);
      CHI_IPC->FreeBuffer(temp_buffer);

      task->output_size_ = expected_size;
      task->decompress_time_ms_ = 0.0;
      task->return_code_ = 0;  // Success (no decompression needed)

      HLOG(kDebug, "GetBlob (no compression): {} bytes", expected_size);
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Decompress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  co_return;
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

chi::u64 Runtime::GetWorkRemaining() const {
  // Return 0 - compressor has no persistent work queue
  return 0;
}

}  // namespace wrp_cte::compressor

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cte::compressor::Runtime)
