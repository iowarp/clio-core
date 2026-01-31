// Copyright 2024 IOWarp contributors
#include <wrp_cte/compressor/compressor_runtime.h>

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
#include "hermes_shm/util/logging.h"
#include "hermes_shm/compress/compress_factory.h"

namespace wrp_cte::compressor {

// Bring chi namespace items into scope for CHI_CUR_WORKER macro
using chi::chi_cur_worker_key_;
using chi::Worker;

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext &ctx) {
  // Note: This is a simplified Create task since we don't have a CreateTask defined
  // In the actual implementation, you would extract config from task parameters

  // Initialize atomic counters
  compression_logical_time_ = 0;

  // Load Q-table model if configured (primary prediction method)
  if (!config_.qtable_model_path_.empty()) {
    try {
      HLOG(kInfo, "Loading Q-table model from: {}", config_.qtable_model_path_);
      qtable_predictor_ = std::make_unique<QTablePredictor>();
      if (qtable_predictor_->Load(config_.qtable_model_path_)) {
        HLOG(kInfo, "Q-table model loaded successfully with {} states",
             qtable_predictor_->GetNumStates());
      } else {
        HLOG(kWarning, "Failed to load Q-table model from: {}", config_.qtable_model_path_);
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
      HLOG(kInfo, "Loading LinReg table model from: {}", config_.linreg_model_path_);
      linreg_predictor_ = std::make_unique<LinRegTablePredictor>();
      if (linreg_predictor_->Load(config_.linreg_model_path_)) {
        HLOG(kInfo, "LinReg table model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load LinReg table model from: {}", config_.linreg_model_path_);
        linreg_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading LinReg table model: {}", e.what());
      linreg_predictor_.reset();
    }
  }

  // Load distribution classifier if configured
  if (!config_.distribution_model_path_.empty()) {
    // Note: DistributionClassifier is template-based - use DistributionClassifierFactory::Classify() directly
    // No model loading needed - the factory uses built-in mathematical classification
    HLOG(kInfo, "Distribution classifier available via factory (no model loading required)");
  }

#ifdef WRP_COMPRESSOR_ENABLE_DENSE_NN
  // Load DNN model weights as fallback if Q-table not available
  if (!qtable_predictor_ && !config_.dnn_model_weights_path_.empty()) {
    try {
      HLOG(kInfo, "Loading DNN model weights from: {}", config_.dnn_model_weights_path_);
      nn_predictor_ = std::make_unique<DenseNNPredictor>();
      if (nn_predictor_->LoadWeights(config_.dnn_model_weights_path_)) {
        HLOG(kInfo, "DNN model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load DNN model weights from: {}", config_.dnn_model_weights_path_);
        nn_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading DNN model: {}", e.what());
      nn_predictor_.reset();
    }
  }
#endif  // WRP_COMPRESSOR_ENABLE_DENSE_NN

  if (!qtable_predictor_ && !linreg_predictor_) {
    HLOG(kDebug, "No compression predictor configured, dynamic compression prediction disabled");
  }

  HLOG(kInfo, "CTE Compressor container created and initialized for pool: {} (ID: {})",
       pool_name_, pool_id_);

  co_return;
}

chi::TaskResume Runtime::Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &ctx) {
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

    HLOG(kInfo, "CTE Compressor container destroyed successfully");
  } catch (const std::exception &e) {
    HLOG(kError, "Exception during compressor destroy: {}", e.what());
  }
  co_return;
}

chi::TaskResume Runtime::Monitor(hipc::FullPtr<MonitorTask> task, chi::RunContext &ctx) {
  // Dynamic schedule: just set pool query
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    // Initialize core client if needed
    if (!core_client_) {
      core_client_ = std::make_unique<wrp_cte::core::Client>(task->core_pool_id_);
    }

    // List all registered targets from core
    auto list_task = core_client_->AsyncListTargets();
    list_task.Wait();

    if (list_task->return_code_ != 0u) {
      HLOG(kWarning, "Failed to list targets from core (error code: {})",
           list_task->return_code_);
      co_return;
    }

    // Update target states for each registered target
    std::lock_guard<std::mutex> lock(target_states_mutex_);

    for (const auto& target_name : list_task->target_names_) {

      // Get detailed info for this target
      auto info_task = core_client_->AsyncGetTargetInfo(target_name);
      info_task.Wait();

      if (info_task->return_code_ == 0u) {
        // Update or create target state entry
        auto& state = target_states_[target_name];
        state.target_name_ = target_name;
        state.target_score_ = info_task->target_score_;
        state.remaining_space_ = info_task->remaining_space_;
        state.bytes_written_ = info_task->bytes_written_;
        state.last_updated_ = std::chrono::steady_clock::now();

        HLOG(kDebug, "Updated target state: {} (score={:.3f}, remaining={} bytes, written={} bytes)",
             target_name, state.target_score_, state.remaining_space_, state.bytes_written_);
      } else {
        HLOG(kWarning, "Failed to get info for target '{}' (error code: {})",
             target_name, info_task->return_code_);
      }
    }

    HLOG(kDebug, "Monitor updated {} target states", target_states_.size());
  } catch (const std::exception& e) {
    HLOG(kError, "Exception during monitor: {}", e.what());
  }
  co_return;
}

// ==============================================================================
// Compression Statistics Estimation
// ==============================================================================

std::vector<CompressionStats> Runtime::EstCompressionStats(
    const void* chunk, chi::u64 chunk_size, const Context& context) {
  std::vector<CompressionStats> results;

  // Calculate compression features from chunk data
  const auto* data = static_cast<const uint8_t*>(chunk);
  chi::u64 sample_size = std::min(chunk_size, static_cast<chi::u64>(65536));

  // Calculate Shannon entropy
  std::vector<int> histogram(256, 0);
  for (chi::u64 i = 0; i < sample_size; ++i) {
    histogram[data[i]]++;
  }
  double entropy = 0.0;
  for (int count : histogram) {
    if (count > 0) {
      double prob = static_cast<double>(count) / static_cast<double>(sample_size);
      entropy -= prob * std::log2(prob);
    }
  }

  // Calculate MAD (Mean Absolute Deviation)
  double mean = 0.0;
  for (chi::u64 i = 0; i < sample_size; ++i) {
    mean += data[i];
  }
  mean /= static_cast<double>(sample_size);
  double mad = 0.0;
  for (chi::u64 i = 0; i < sample_size; ++i) {
    mad += std::abs(static_cast<double>(data[i]) - mean);
  }
  mad /= static_cast<double>(sample_size);

  // Calculate second derivative mean (curvature)
  double second_deriv_sum = 0.0;
  chi::u64 deriv_count = 0;
  for (chi::u64 i = 1; i < sample_size - 1 && i < 999; ++i) {
    double second_deriv = static_cast<double>(data[i + 1]) -
                          2.0 * static_cast<double>(data[i]) +
                          static_cast<double>(data[i - 1]);
    second_deriv_sum += std::abs(second_deriv);
    deriv_count++;
  }
  double second_derivative_mean = (deriv_count > 0) ?
      (second_deriv_sum / static_cast<double>(deriv_count)) : 0.0;

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
    if (context.target_psnr_ > 0 && pred.psnr_db > 0 && pred.psnr_db < context.target_psnr_) {
      continue;
    }

    // Add to results with library and preset
    results.emplace_back(lib_id, config_id, pred.compression_ratio,
                         pred.compression_time_ms, pred.compression_time_ms,
                         pred.psnr_db);
  }

  return results;
}

double Runtime::EstWorkflowCompressTime(
    chi::u64 chunk_size, double tier_bw, const CompressionStats& stats,
    const Context& context) {

  double compressed_size = chunk_size / stats.compression_ratio_;
  double transfer_time_ms = (compressed_size / tier_bw) * 1000.0;

  if (stats.psnr_db_ == 0.0) {
    // Lossless compression
    return stats.compress_time_ms_ + stats.decompress_time_ms_ + transfer_time_ms;
  } else {
    // Lossy compression - may need verification decompression
    double psnr_check_prob = static_cast<double>(context.psnr_chance_) / 100.0;
    return stats.compress_time_ms_ +
           (1.0 + psnr_check_prob) * stats.decompress_time_ms_ +
           transfer_time_ms;
  }
}

std::tuple<int, int, int, double> Runtime::BestCompressRatio(
    const void* chunk, chi::u64 chunk_size, int container_id,
    const std::vector<CompressionStats>& stats, const Context& context) {

  int best_tier = 0;
  int best_lib = 0;
  int best_preset = 2;  // Default: BALANCED
  double best_time = std::numeric_limits<double>::max();
  double best_ratio = 1.0;

  // Get target bandwidth from cached target states
  double tier_bw = 1e9;  // Default: 1 GB/s
  {
    std::lock_guard<std::mutex> lock(target_states_mutex_);
    if (!target_states_.empty()) {
      // Find target with highest score (best performance)
      float max_score = 0.0f;
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > max_score) {
          max_score = state.target_score_;
          // Estimate bandwidth from normalized log score
          // score = log(bw+1) / log(1000+1), solve for bw
          tier_bw = std::pow(1001.0, max_score) - 1.0;
          tier_bw = std::max(tier_bw, 1e6);  // At least 1 MB/s
          tier_bw = std::min(tier_bw, 1e10);  // Cap at 10 GB/s
        }
      }
    }
  }

  for (const auto& stat : stats) {
    // Calculate workflow time for this compression
    double est_time = EstWorkflowCompressTime(chunk_size, tier_bw, stat, context);

    // Choose compression with best ratio that meets time constraints
    if (stat.compression_ratio_ > best_ratio) {
      best_ratio = stat.compression_ratio_;
      best_lib = stat.compress_lib_;
      best_preset = stat.compress_preset_;
      best_time = est_time;
      best_tier = 0;
    }
  }

  return std::make_tuple(best_tier, best_lib, best_preset, best_time);
}

std::tuple<int, int, int, double> Runtime::BestCompressTime(
    const void* chunk, chi::u64 chunk_size, int container_id,
    const std::vector<CompressionStats>& stats, const Context& context) {

  int best_tier = 0;
  int best_lib = 0;
  int best_preset = 2;  // Default: BALANCED
  double best_time = std::numeric_limits<double>::max();

  // Get target bandwidth from cached target states
  double tier_bw = 1e9;  // Default: 1 GB/s
  {
    std::lock_guard<std::mutex> lock(target_states_mutex_);
    if (!target_states_.empty()) {
      // Find target with highest score (best performance)
      float max_score = 0.0f;
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > max_score) {
          max_score = state.target_score_;
          // Estimate bandwidth from normalized log score
          // score = log(bw+1) / log(1000+1), solve for bw
          tier_bw = std::pow(1001.0, max_score) - 1.0;
          tier_bw = std::max(tier_bw, 1e6);  // At least 1 MB/s
          tier_bw = std::min(tier_bw, 1e10);  // Cap at 10 GB/s
        }
      }
    }
  }

  // For each compression library and tier, calculate workflow time
  for (const auto& stat : stats) {
    double est_time = EstWorkflowCompressTime(chunk_size, tier_bw, stat, context);

    // Choose combination with best performance
    if (est_time < best_time) {
      best_time = est_time;
      best_lib = stat.compress_lib_;
      best_preset = stat.compress_preset_;
      best_tier = 0;
    }
  }

  return std::make_tuple(best_tier, best_lib, best_preset, best_time);
}

std::tuple<int, int, int, double> Runtime::BestCompressForNode(
    const Context& context, const void* chunk, chi::u64 chunk_size,
    int container_id, const std::vector<CompressionStats>& stats) {

  // Choose strategy based on context objective
  if (context.max_performance_) {
    // Objective: minimize time
    return BestCompressTime(chunk, chunk_size, container_id, stats, context);
  } else {
    // Objective: maximize compression ratio
    return BestCompressRatio(chunk, chunk_size, container_id, stats, context);
  }
}

// ==============================================================================
// Task Execution Methods
// ==============================================================================

// Static atomic trace key counter for generating unique trace IDs
static std::atomic<chi::u64> g_trace_key_counter{1};

// Helper function to write trace log entry
static void WriteTraceLog(const std::string& trace_folder, const std::string& log_name,
                          chi::u32 container_id, const std::string& entry) {
  if (trace_folder.empty()) return;

  try {
    std::string log_path = trace_folder + "/" + log_name + "." + std::to_string(container_id);
    std::ofstream log_file(log_path, std::ios::app);
    if (log_file.is_open()) {
      log_file << entry << std::endl;
      log_file.close();
    }
  } catch (const std::exception& e) {
    HLOG(kWarning, "Failed to write trace log: {}", e.what());
  }
}

chi::TaskResume Runtime::DynamicSchedule(hipc::FullPtr<DynamicScheduleTask> task,
                                         chi::RunContext& ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    // Extract task parameters
    chi::u64 chunk_size = task->chunk_size_;
    void* chunk_data = task->chunk_data_;
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
        log_entry << context.trace_key_ << ","
                  << stat.compress_lib_ << ","
                  << stat.compression_ratio_ << ","
                  << stat.compress_time_ms_ << ","
                  << stat.decompress_time_ms_ << ","
                  << stat.psnr_db_;
        WriteTraceLog(config_.trace_folder_path_, "predicted_stats.log",
                      pool_id_.major_, log_entry.str());
      }
    }

    // Choose best compression strategy
    // For simplicity, use container_id = 0 (can be enhanced to use actual container ID)
    auto [best_tier, best_lib, best_preset, best_time] = BestCompressForNode(context, chunk_data,
                                                                               chunk_size, 0, stats);

    // Update context with selected compression library and preset
    context.compress_lib_ = best_lib;
    context.compress_preset_ = best_preset;

    // Log scheduling decision time if tracing enabled
    if (context.trace_) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

      std::ostringstream log_entry;
      log_entry << context.trace_key_ << "," << duration_ms;
      WriteTraceLog(config_.trace_folder_path_, "sched_decision.log",
                    pool_id_.major_, log_entry.str());
    }

    task->return_code_ = 0;
  } catch (const std::exception& e) {
    HLOG(kError, "Exception in DynamicSchedule: {}", e.what());
    task->return_code_ = 1;
  }

  co_return;
}

chi::TaskResume Runtime::Compress(hipc::FullPtr<CompressTask> task,
                                  chi::RunContext& ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    // Extract task parameters
    void* input_data = task->input_data_;
    chi::u64 input_size = task->input_size_;
    Context& context = task->context_;

    // Validate inputs
    if (input_data == nullptr || input_size == 0) {
      task->return_code_ = 1;  // Invalid input
      co_return;
    }

    if (context.compress_lib_ <= 0) {
      task->return_code_ = 2;  // No compression library specified
      co_return;
    }

    // Map compress_lib_ ID to library name
    const char* lib_names[] = {"brotli", "bzip2", "blosc2", "fpzip", "lz4",
                                "lzma", "snappy", "sz3", "zfp", "zlib", "zstd"};
    std::string library_name = (context.compress_lib_ >= 0 && context.compress_lib_ <= 10) ?
                               lib_names[context.compress_lib_] : "zstd";

    // Map preset integer to enum
    hshm::CompressionPreset preset = hshm::CompressionPreset::BALANCED;
    if (context.compress_preset_ == 1) preset = hshm::CompressionPreset::FAST;
    else if (context.compress_preset_ == 2) preset = hshm::CompressionPreset::BALANCED;
    else if (context.compress_preset_ == 3) preset = hshm::CompressionPreset::BEST;

    // Create compressor with specified preset
    auto compressor = hshm::CompressionFactory::GetPreset(library_name, preset);

    if (!compressor) {
      HLOG(kWarning, "Failed to create compressor for library: {}", library_name);
      task->return_code_ = 3;  // Compressor creation failed
      co_return;
    }

    auto compress_start = std::chrono::high_resolution_clock::now();

    // Allocate buffer for compressed data (worst case: original size + 5% overhead)
    std::vector<char> compressed_buffer(input_size + (input_size / 20) + 1024);

    // Compress the data
    size_t compressed_size = compressed_buffer.size();
    bool success = compressor->Compress(compressed_buffer.data(), compressed_size,
                                        reinterpret_cast<char*>(input_data), input_size);

    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();

    if (success && compressed_size < input_size) {
      // Compression succeeded and reduced size
      compressed_buffer.resize(compressed_size);

      // Allocate output buffer (caller will free this)
      task->output_data_ = malloc(compressed_size);
      if (task->output_data_ == nullptr) {
        task->return_code_ = 4;  // Memory allocation failed
        co_return;
      }

      // Copy compressed data to output
      std::memcpy(task->output_data_, compressed_buffer.data(), compressed_size);
      task->output_size_ = compressed_size;

      // Log compression telemetry
      CompressionTelemetry telemetry(
          CteOp::kPutBlob, context.compress_lib_, input_size, compressed_size,
          compress_time, 0.0, 0.0, std::chrono::steady_clock::now(),
          compression_logical_time_.fetch_add(1));
      LogCompressionTelemetry(telemetry);

      HLOG(kDebug, "Compression: {} bytes -> {} bytes (ratio: {:.2f}, time: {:.2f}ms)",
           input_size, compressed_size, static_cast<double>(input_size) / compressed_size,
           compress_time);

      // Update target capacity in cache (optimistic update)
      {
        std::lock_guard<std::mutex> lock(target_states_mutex_);
        if (!target_states_.empty()) {
          // Find target with highest score (most likely to be used for placement)
          std::string best_target_name;
          float max_score = -1.0f;
          for (const auto& [name, state] : target_states_) {
            if (state.target_score_ > max_score) {
              max_score = state.target_score_;
              best_target_name = name;
            }
          }

          // Update the best target's capacity
          if (!best_target_name.empty()) {
            auto& target = target_states_[best_target_name];
            if (target.remaining_space_ >= compressed_size) {
              target.remaining_space_ -= compressed_size;
              target.bytes_written_ += compressed_size;
              HLOG(kDebug, "Updated target '{}' capacity: -{} bytes (remaining: {} bytes)",
                   best_target_name, compressed_size, target.remaining_space_);
            }
          }
        }
      }

      task->return_code_ = 0;  // Success
    } else {
      // Compression failed or didn't reduce size
      HLOG(kDebug, "Compression not beneficial, returning original data");
      task->output_data_ = nullptr;
      task->output_size_ = input_size;
      task->return_code_ = 5;  // Compression not beneficial
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Compress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  co_return;
}

chi::TaskResume Runtime::Decompress(hipc::FullPtr<DecompressTask> task,
                                    chi::RunContext& ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    // Extract task parameters
    void* input_data = task->input_data_;
    chi::u64 input_size = task->input_size_;
    chi::u64 expected_output_size = task->expected_output_size_;
    int compress_lib = task->compress_lib_;
    int compress_preset = task->compress_preset_;

    // Validate inputs
    if (input_data == nullptr || input_size == 0) {
      task->return_code_ = 1;  // Invalid input
      co_return;
    }

    if (compress_lib <= 0) {
      task->return_code_ = 2;  // No compression library specified
      co_return;
    }

    // Map compress_lib ID to library name
    const char* lib_names[] = {"brotli", "bzip2", "blosc2", "fpzip", "lz4",
                                "lzma", "snappy", "sz3", "zfp", "zlib", "zstd"};
    std::string library_name = (compress_lib >= 0 && compress_lib <= 10) ?
                               lib_names[compress_lib] : "zstd";

    // Map preset integer to enum
    hshm::CompressionPreset preset = hshm::CompressionPreset::BALANCED;
    if (compress_preset == 1) preset = hshm::CompressionPreset::FAST;
    else if (compress_preset == 2) preset = hshm::CompressionPreset::BALANCED;
    else if (compress_preset == 3) preset = hshm::CompressionPreset::BEST;

    // Create decompressor with same preset that was used for compression
    auto decompressor = hshm::CompressionFactory::GetPreset(library_name, preset);

    if (!decompressor) {
      HLOG(kWarning, "Failed to create decompressor for library: {}", library_name);
      task->return_code_ = 3;  // Decompressor creation failed
      co_return;
    }

    auto decompress_start = std::chrono::high_resolution_clock::now();

    // Allocate buffer for decompressed data
    std::vector<char> decompressed_buffer(expected_output_size);

    // Decompress the data
    size_t decompressed_size = decompressed_buffer.size();
    bool success = decompressor->Decompress(decompressed_buffer.data(), decompressed_size,
                                            reinterpret_cast<char*>(input_data), input_size);

    auto decompress_end = std::chrono::high_resolution_clock::now();
    double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();

    if (success) {
      // Decompression succeeded
      // Allocate output buffer (caller will free this)
      task->output_data_ = malloc(decompressed_size);
      if (task->output_data_ == nullptr) {
        task->return_code_ = 4;  // Memory allocation failed
        co_return;
      }

      // Copy decompressed data to output
      std::memcpy(task->output_data_, decompressed_buffer.data(), decompressed_size);
      task->output_size_ = decompressed_size;

      // Log decompression telemetry
      CompressionTelemetry telemetry(
          CteOp::kGetBlob, compress_lib, decompressed_size, input_size,
          0.0, decompress_time, 0.0, std::chrono::steady_clock::now(),
          compression_logical_time_.fetch_add(1));
      LogCompressionTelemetry(telemetry);

      HLOG(kDebug, "Decompression: {} bytes -> {} bytes (time: {:.2f}ms)",
           input_size, decompressed_size, decompress_time);

      task->return_code_ = 0;  // Success
    } else {
      HLOG(kError, "Decompression failed");
      task->output_data_ = nullptr;
      task->output_size_ = 0;
      task->return_code_ = 5;  // Decompression failed
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
    log_entry << telemetry.logical_time_ << ","
              << telemetry.compress_lib_ << ","
              << telemetry.original_size_ << ","
              << telemetry.compressed_size_ << ","
              << telemetry.compress_time_ms_ << ","
              << telemetry.decompress_time_ms_ << ","
              << telemetry.psnr_db_;

    std::string log_name = (telemetry.op_ == CteOp::kPutBlob) ?
                           "compress_stats.log" : "decompress_stats.log";
    WriteTraceLog(config_.trace_folder_path_, log_name, pool_id_.major_, log_entry.str());
  }
}

chi::u64 Runtime::GetWorkRemaining() const {
  // Return 0 - compressor has no persistent work queue
  return 0;
}

} // namespace wrp_cte::compressor

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cte::compressor::Runtime)
