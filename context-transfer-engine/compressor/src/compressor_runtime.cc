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
#include "clio_ctp/util/gpu_api.h"
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
#include <cuda.h>
#include <cuda_runtime.h>

// In-process blob locator, registered by the CTE core. Lets the compressor
// find where a blob physically lives, which is how a page already resident on
// a DEVICE tier is decoded straight out of that tier instead of being fetched
// into scratch first.
extern "C" int clio_cte_locate(const void *tag_id, const char *name,
                               unsigned long long *pool_u64,
                               unsigned long long *target_off,
                               unsigned long long *stored_size);
extern "C" char *clio_direct_dev_base(unsigned long long pool_id);
#if CTP_ENABLE_NVCOMP
// The C batched entry points, from the ORDINARY host library. Every algorithm
// exposes the same eleven-parameter signature, which is what lets one dispatch
// cover all of them -- and none of it needs nvcomp's device API.
#include <nvcomp/ans.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/gzip.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>
#endif
#endif
#include "clio_ctp/compress/data_stats.h"
#include "clio_ctp/util/logging.h"

namespace clio::cte::compressor {

/** Waiter poll interval and cap for the batched GPU decompress path. 20us x
 *  5000 = 100 ms: far longer than a batch needs, far shorter than a hang. */
static constexpr double kDecompWaitPollUs = 20.0;
/**
 * The wait must cover QUEUEING, not just one batch.
 *
 * A request that arrives just after a batch starts waits for that batch, then
 * the linger, then its own batch -- with 50-70 ms batches that is ~140 ms, and
 * a 100 ms budget timed out reliably. The bound exists only so a wedged
 * drainer cannot hang a worker forever, so it should be generous: 2 s.
 */
static constexpr int kDecompWaitMaxSpins = 100000;

// Bring chi namespace items into scope for CLIO_CUR_WORKER macro
using clio::run::chi_cur_worker_key_;
using clio::run::Worker;

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
  /**
   * EXACT compressed payload size (bytes following this header).
   *
   * Frame-exact codecs need it. A reader does not know the compressed length
   * a priori, so it over-allocates its fetch and the trailing bytes are
   * garbage; passing that over-estimate to LZ4_decompress_safe (or zstd) makes
   * decompression FAIL. Without this field the reader could only guess
   * "request size minus header", which is exactly that over-estimate.
   *
   * 0 means "written before this field existed" -- readers fall back to the
   * derived estimate for those blobs.
   */
  uint64_t compressed_size_;

  CompressionHeader()
      : magic_(kMagic),
        compress_lib_(0),
        compress_preset_(0),
        original_size_(0),
        compressed_size_(0) {}

  CompressionHeader(uint32_t lib, uint32_t preset, uint64_t orig_size,
                    uint64_t comp_size = 0)
      : magic_(kMagic),
        compress_lib_(lib),
        compress_preset_(preset),
        original_size_(orig_size),
        compressed_size_(comp_size) {}

  bool IsValid() const { return magic_ == kMagic; }
};
static_assert(sizeof(CompressionHeader) == 32,
              "CompressionHeader must be 32 bytes");

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Load configuration from compose YAML (or direct CreateParams)
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target

  // Initialize the core client using next_pool_id from compose
  if (!config_.next_pool_id_.IsNull()) {
    core_client_ = std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }

  // One GPU stream for every GPU codec this module hands out, created HERE --
  // once, at module creation -- and never inside an operation.
  //
  // A codec that creates its own stream deadlocks the device: cudaStreamCreate
  // blocks while a kernel is resident, and the kernels a GPU codec serves are
  // precisely the ones suspended waiting for the compress or decompress being
  // set up. Creating it at Create time is also simply where it belongs: stream
  // creation is expensive and its cost has nothing to do with any one blob.
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
  // A GPU codec also needs its OWN CUDA context (see the header): it cannot run
  // in the context whose kernel is spinning on the fault. InitCodecContext
  // creates that context, its buffers, and the stream INSIDE it.
  if (codec_ctx_ == nullptr) {
    const bool _ok = InitCodecContext();
    if (getenv("CLIO_CODEC_TRACE")) fprintf(stderr, "[TRACE] InitCodecContext -> %d\n", (int)_ok);
    if (_ok) {
      HLOG(kInfo,
           "compressor: GPU codec context ready ({} MB device buffers)",
           codec_buf_bytes_ >> 20);
    } else {
      DestroyCodecContext();
      HLOG(kWarning,
           "compressor: GPU codec context unavailable; GPU codecs fall back "
           "to their CPU equivalent");
    }
  }
#elif CTP_ENABLE_GPU
  if (gpu_stream_ == nullptr) {
    gpu_stream_ = ctp::GpuApi::CreateStream();
    ctp::CompressionFactory::SetGpuStream(gpu_stream_);
  }
#endif

#if CTP_ENABLE_GPU
  // One device allocation for the compressed side of GPU codec operations.
  // See the header for why this cannot be done per operation.
  if (gpu_scratch_base_ == nullptr) {
    size_t total_bytes = 128ULL << 20;
    if (const char *e = std::getenv("CLIO_COMPRESS_GPU_SCRATCH_MB")) {
      const long mb = std::atol(e);
      if (mb > 0) total_bytes = static_cast<size_t>(mb) << 20;
    }
    constexpr size_t kSlabs = 16;
    gpu_scratch_slab_ = total_bytes / kSlabs;
    // The compressed bytes are handed straight to PutBlob and written to the
    // device tier from here, so this stays plain device memory -- see
    // ScratchShmPtr for how a raw device pointer is addressed.
    gpu_scratch_base_ = ctp::GpuApi::Malloc<char>(total_bytes);
    if (gpu_scratch_base_ != nullptr) {
      gpu_scratch_free_.reserve(kSlabs);
      for (size_t i = 0; i < kSlabs; ++i) {
        gpu_scratch_free_.push_back(gpu_scratch_base_ + i * gpu_scratch_slab_);
      }
      HLOG(kInfo, "compressor: {} MB device scratch ({} slabs of {} KB)",
           total_bytes >> 20, kSlabs, gpu_scratch_slab_ >> 10);
    } else {
      gpu_scratch_slab_ = 0;
      HLOG(kWarning,
           "compressor: device scratch allocation failed -- GPU codecs will "
           "use the host path");
    }
  }
#endif

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

  if (!qtable_predictor_ && !linreg_predictor_) {
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
    const void* chunk, clio::run::u64 chunk_size, const Context& context) {
  std::vector<CompressionStats> results;

  // Determine data type from context
  // context.data_type_: 0 = char/uint8, 1 = float
  ctp::DataType data_type = (context.data_type_ == 1) ? ctp::DataType::FLOAT32
                                                       : ctp::DataType::UINT8;
  size_t type_size = ctp::DataStatisticsFactory::GetTypeSize(data_type);

  // Calculate number of elements (sample up to 64KB for efficiency)
  clio::run::u64 sample_bytes = std::min(chunk_size, static_cast<clio::run::u64>(65536));
  size_t num_elements = sample_bytes / type_size;
  if (num_elements == 0) {
    num_elements = 1;
  }

  // Calculate compression features using DataStatisticsFactory
  double entropy = ctp::DataStatisticsFactory::CalculateShannonEntropy(
      chunk, num_elements, data_type);
  double mad =
      ctp::DataStatisticsFactory::CalculateMAD(chunk, num_elements, data_type);
  double second_derivative_mean =
      ctp::DataStatisticsFactory::CalculateSecondDerivative(
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
    auto stats = EstCompressionStats(chunk_data, chunk_size, context);

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
        clio::run::PoolQuery::Local(), task->tag_id_, task->blob_name_.str(),
        task->offset_, task->size_, task->blob_data_, task->score_, context,
        task->flags_, task->core_pool_id_);
    CLIO_CO_AWAIT(compress_task);

    // Copy results back
    task->context_ = compress_task->context_;
    task->tier_score_ = compress_task->tier_score_;
    task->return_code_ = compress_task->return_code_;

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
    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (context.compress_preset_ == 1) {
      preset = ctp::CompressionPreset::FAST;
    } else if (context.compress_preset_ == 3) {
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

    auto compress_start = std::chrono::high_resolution_clock::now();

    // Allocate buffer for compressed data (worst case: original size + 5%
    // overhead)
    std::vector<char> compressed_buffer(input_size + (input_size / 20) + 1024);

    // Compress the data
    size_t compressed_size = compressed_buffer.size();
    // Convert ShmPtr to raw pointer via FullPtr
    auto input_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
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

    // Same meaningful-gain bar as CompressIntoShm (see there): marginal
    // compression charges every reader a decompress for ~no capacity.
    if (success && total_stored_size < input_size - input_size / 8) {
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
      auto compressed_shm = CLIO_IPC->AllocateBuffer(total_stored_size);
      if (compressed_shm.IsNull()) {
        HLOG(kError, "Failed to allocate shared memory for compressed data");
        task->return_code_ = 4;  // Memory allocation failed
        CLIO_CO_RETURN;
      }

      // Write compression header
      CompressionHeader header(context.compress_lib_, context.compress_preset_,
                               input_size, compressed_size);
      std::memcpy(compressed_shm.ptr_, &header, header_size);

      // Write compressed data after header
      std::memcpy(compressed_shm.ptr_ + header_size, compressed_buffer.data(),
                  compressed_size);

      // Tell the runtime these bytes are no longer the caller's bytes, so it
      // can mark the blob authoritatively (issue #818). This is the ONLY place
      // that knows it for certain -- compress_lib_ is set on the not-beneficial
      // path below too, where the stored bytes are raw.
      context.transform_flags_ |= clio::cte::core::kBlobTransformed |
                                  clio::cte::core::kBlobTransformCompressed;

      // Call PutBlob with header + compressed data
      ctp::ipc::ShmPtr<> compressed_shm_ptr =
          compressed_shm.shm_.template Cast<void>();
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_,
          total_stored_size, compressed_shm_ptr, task->score_, context,
          task->flags_, clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put_task);

      // Free compressed data buffer
      CLIO_IPC->FreeBuffer(compressed_shm);

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

    // Extract task parameters (same as GetBlobTask)
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

    // Allocate temporary buffer to receive compressed data from GetBlob
    // We don't know the compressed size, so allocate expected_size as upper
    // bound
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
      int compress_preset = static_cast<int>(header->compress_preset_);
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

      // Get compressed data (after header).
      //
      // Use the EXACT payload length the writer recorded. Deriving it from the
      // request size minus the header is an OVER-ESTIMATE: the reader does not
      // know the compressed length a priori and over-allocates its fetch, so
      // the trailing bytes are garbage. Frame-exact codecs (lz4, zstd) then
      // fail on the bogus length -- which is exactly what
      // CompressionHeader::compressed_size_ was added to carry. Fall back to
      // the derived value for blobs written before that field existed
      // (compressed_size_ == 0).
      char* compressed_data = temp_buffer.ptr_ + header_size;
      size_t compressed_size = (header->compressed_size_ != 0)
          ? static_cast<size_t>(header->compressed_size_)
          : (expected_size - header_size);

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

#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
namespace {
/** Make the codec context current for a scope, then restore the caller's. */
class CodecCtxGuard {
 public:
  explicit CodecCtxGuard(void *ctx) : ok_(false) {
    ok_ = (cuCtxPushCurrent(static_cast<CUcontext>(ctx)) == CUDA_SUCCESS);
  }
  ~CodecCtxGuard() {
    if (ok_) {
      CUcontext popped = nullptr;
      cuCtxPopCurrent(&popped);
    }
  }
  bool ok() const { return ok_; }

 private:
  bool ok_;
};
}  // namespace

bool Runtime::InitCodecContext() {
  // Set CLIO_COMPRESS_GPU_CODEC=0 to force GPU codecs onto the CPU path.
  //
  // On by default: a GPU codec exists so that a compressed page is decompressed
  // where it is needed, straight into the faulting page, with nothing staged
  // through the host. Falling back to the CPU codec throws that away -- it
  // stages the page out, decompresses, and stages a full uncompressed page
  // back in, which is strictly more work than not compressing at all.
  const char *opt_out = std::getenv("CLIO_COMPRESS_GPU_CODEC");
  if (opt_out != nullptr && opt_out[0] == '0') {
    return false;
  }
  if (cuInit(0) != CUDA_SUCCESS) {
    return false;
  }
  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    return false;
  }
  // Remember whichever context the faulting side uses; peer copies name it
  // explicitly. cudaFree(0) forces the runtime's primary context to exist so
  // there is something to record.
  cudaFree(nullptr);
  CUcontext primary = nullptr;
  if (cuCtxGetCurrent(&primary) != CUDA_SUCCESS || primary == nullptr) {
    if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS) {
      return false;
    }
  }
  primary_ctx_ = primary;

  CUcontext ctx = nullptr;
  // cuCtxCreate is an API-versioned macro and CUDA 13 repointed it: 12.x maps
  // it to cuCtxCreate_v2(ctx, flags, dev), 13.x to cuCtxCreate_v4(ctx,
  // CUctxCreateParams*, flags, dev). Writing only the v4 form made this file
  // fail to compile against a 12.x toolkit ("cannot convert std::nullptr_t to
  // unsigned int"), which is easy to miss because the toolkit that supplies
  // the headers need not be the one on /usr/local/cuda.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
  const CUresult ctx_rc = cuCtxCreate(&ctx, nullptr, 0, dev);
#else
  const CUresult ctx_rc = cuCtxCreate(&ctx, 0, dev);
#endif
  if (ctx_rc != CUDA_SUCCESS) {
    return false;
  }
  // cuCtxCreate leaves the new context CURRENT. Everything after Create runs
  // on the faulting side, so put its context back immediately.
  CUcontext popped = nullptr;
  cuCtxPopCurrent(&popped);
  codec_ctx_ = ctx;

  size_t buf = 8ULL << 20;
  if (const char *e = std::getenv("CLIO_COMPRESS_GPU_BUF_MB")) {
    const long mb = std::atol(e);
    if (mb > 0) buf = static_cast<size_t>(mb) << 20;
  }
  size_t slots = 16;
  if (const char *e = std::getenv("CLIO_COMPRESS_GPU_SLOTS")) {
    const long n = std::atol(e);
    if (n > 0) slots = static_cast<size_t>(n);
  }
  CodecCtxGuard guard(codec_ctx_);
  if (!guard.ok()) {
    return false;
  }
  codec_slots_.resize(slots);
  for (size_t i = 0; i < slots; ++i) {
    CUdeviceptr d = 0;
    if (cuMemAlloc(&d, buf) != CUDA_SUCCESS) {
      return false;
    }
    codec_slots_[i].buf = reinterpret_cast<void *>(d);
    CUdeviceptr o = 0;
    if (cuMemAlloc(&o, buf) != CUDA_SUCCESS) {
      return false;
    }
    codec_slots_[i].obuf = reinterpret_cast<void *>(o);
    // A stream PER SLOT. Sharing one stream would serialize the concurrent
    // operations this pool exists to allow, since each waits on its own work.
    codec_slots_[i].stream = ctp::GpuApi::CreateStream();
    codec_free_.push_back(i);
  }
  codec_buf_bytes_ = buf;
  // The factory's stream is only a default for codecs built outside this pool.
  gpu_stream_ = codec_slots_[0].stream;
  ctp::CompressionFactory::SetGpuStream(gpu_stream_);
  // Batched decompression is opt-in while it is brought up; the synchronous
  // per-fault path stays the default.
  // ON by default. Batching is not an experiment any more: it is the only
  // shape that launches one kernel for N pages instead of N kernels each
  // awaited. CLIO_COMPRESS_GPU_BATCH=0 disables it for A/B measurement.
  const char *batch_env = std::getenv("CLIO_COMPRESS_GPU_BATCH");
  batch_enabled_ = (batch_env == nullptr || batch_env[0] != '0');
  if (batch_enabled_) {
    batch_stop_.store(false, std::memory_order_release);
    batch_thread_ = std::thread([this]() { BatchDrainLoop(); });
    HLOG(kInfo, "compressor: batched GPU decompress enabled (one launch per drain)");
  }
  return true;
}

void Runtime::RunDecompBatch(std::vector<std::shared_ptr<PendingDecomp>> &batch) {
  if (batch.empty()) return;
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
  // Build ONE batch covering every pending page and hand it to nvcomp in a
  // single launch. The old shape here decompressed page by page and
  // synchronized after each, which is what made a spilled page cost ~1.6ms:
  // the manager API synchronizes twice per call (create_manager parses the
  // header on the host, Decompress waits at the end), so N pages meant 2N
  // serialization points however they were queued.
  std::vector<DecompItem> items;
  std::vector<size_t> idx;
  items.reserve(batch.size());
  idx.reserve(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    auto &p = batch[i];
    if (p->abandoned.load(std::memory_order_acquire)) continue;
    if (p->src_device == nullptr) continue;   // host-sourced: not this path
    DecompItem it;
    it.src_device = p->src_device;
    it.stored_size = p->stored_size;
    it.dst_device = p->dst;
    it.dst_bytes = p->dst_bytes;
    items.push_back(it);
    idx.push_back(i);
  }
  if (!items.empty()) {
    std::vector<char> ok;
    const size_t n = GpuDecompressBatch(items, &ok);
    for (size_t k = 0; k < idx.size(); ++k) {
      batch[idx[k]]->ok = (k < ok.size() && ok[k] != 0);
    }
    if (getenv("CLIO_CODEC_TRACE")) {
      fprintf(stderr, "[DRAIN] batch=%zu decoded=%zu in ONE launch\n",
              items.size(), n);
      fflush(stderr);
    }
  }
#endif
  // Publish LAST: a waiter may be freed the instant it observes done.
  for (auto &p : batch) p->done.store(true, std::memory_order_release);
}

void Runtime::BatchDrainLoop() {
  // Linger before draining. Taking the queue the moment it is non-empty yields
  // batches of one, and a batch of one costs what no batching costs -- the
  // context entry is charged per ENTRY, not per page. Bounded both ways: stop
  // early once enough have gathered, give up after kLingerMaxUs so a lone
  // fault is not held hostage to peers that never arrive.
  constexpr int kLingerStepUs = 50;
  int kLingerMaxUs = 400;
  if (const char *e = std::getenv("CLIO_COMPRESS_LINGER_US")) {
    const int v = std::atoi(e);
    if (v > 0) kLingerMaxUs = v;
  }
  // How many pages to gather before launching. NOT codec_slots_.size(): the
  // batched decompressor does not take a codec slot per page -- it builds its
  // own descriptor arrays and hands nvcomp every chunk in one call -- so the
  // slot count was an unrelated cap that held batches to 8-16. The linger
  // bound below still stops a lone fault waiting on peers that never come.
  size_t target = 128;
  if (const char *e = std::getenv("CLIO_COMPRESS_BATCH_TARGET")) {
    const long v = std::atol(e);
    if (v > 0) target = static_cast<size_t>(v);
  }
  while (!batch_stop_.load(std::memory_order_acquire)) {
    size_t queued = 0;
    {
      std::lock_guard<std::mutex> g(batch_mu_);
      queued = batch_.size();
    }
    if (queued == 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(kLingerStepUs));
      continue;
    }
    for (int waited = 0; queued < target && waited < kLingerMaxUs;
         waited += kLingerStepUs) {
      std::this_thread::sleep_for(std::chrono::microseconds(kLingerStepUs));
      std::lock_guard<std::mutex> g(batch_mu_);
      queued = batch_.size();
    }
    std::vector<std::shared_ptr<PendingDecomp>> mine;
    {
      std::lock_guard<std::mutex> g(batch_mu_);
      mine.swap(batch_);
    }
    if (mine.empty()) continue;
    const bool trace = getenv("CLIO_CODEC_TRACE") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    RunDecompBatch(mine);
    if (trace) {
      size_t nok = 0;
      for (auto &p : mine) nok += p->ok ? 1 : 0;
      fprintf(stderr, "[DRAIN] batch=%zu ok=%zu in %.1f ms\n", mine.size(), nok,
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count());
      fflush(stderr);
    }
  }
}

size_t Runtime::AcquireCodecSlot() {
  std::lock_guard<std::mutex> guard(codec_mu_);
  if (codec_free_.empty()) {
    return SIZE_MAX;
  }
  const size_t idx = codec_free_.back();
  codec_free_.pop_back();
  return idx;
}

void Runtime::ReleaseCodecSlot(size_t idx) {
  if (idx == SIZE_MAX) {
    return;
  }
  std::lock_guard<std::mutex> guard(codec_mu_);
  codec_free_.push_back(idx);
}

Runtime::~Runtime() {
  batch_stop_.store(true, std::memory_order_release);
  if (batch_thread_.joinable()) {
    batch_thread_.join();
  }
}

void Runtime::DestroyCodecContext() {
  if (codec_ctx_ == nullptr) {
    return;
  }
  batch_stop_.store(true, std::memory_order_release);
  if (batch_thread_.joinable()) batch_thread_.join();
  {
    CodecCtxGuard guard(codec_ctx_);
    if (guard.ok()) {
      for (auto &slot : codec_slots_) {
        if (slot.buf != nullptr) {
          cuMemFree(reinterpret_cast<CUdeviceptr>(slot.buf));
        }
        if (slot.obuf != nullptr) {
          cuMemFree(reinterpret_cast<CUdeviceptr>(slot.obuf));
        }
      }
    }
  }
  codec_slots_.clear();
  codec_free_.clear();
  cuCtxDestroy(static_cast<CUcontext>(codec_ctx_));
  codec_ctx_ = nullptr;
  codec_buf_bytes_ = 0;
  gpu_stream_ = nullptr;
}

/**
 * Decompress a stored blob that is ALREADY in device memory, into device
 * memory. Device pointer in, device pointer out, and no copy of the payload
 * in either direction -- nvcomp reads the compressed bytes where they lie.
 *
 * The host-source variant below has to stage the payload H2D into a codec
 * slot first, and the caller had to bring it D2H out of the tier to call it
 * at all. For a kHBM tier that is a device->host->device round trip of the
 * compressed bytes on every read.
 *
 * @param stored_device  device pointer to [CompressionHeader][codec bytes]
 * @param header         host copy of that header (32 bytes of metadata; the
 *                       host cannot read the device copy directly)
 */
namespace {

/** Small device array built from a host vector; frees itself. */
template <typename T>
class DeviceArray {
 public:
  DeviceArray(const std::vector<T> &h, cudaStream_t stream) : n_(h.size()) {
    if (n_ == 0) { ok_ = true; return; }
    if (cudaMalloc(&d_, n_ * sizeof(T)) != cudaSuccess) return;
    ok_ = cudaMemcpyAsync(d_, h.data(), n_ * sizeof(T),
                          cudaMemcpyHostToDevice, stream) == cudaSuccess;
  }
  ~DeviceArray() { if (d_ != nullptr) cudaFree(d_); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  bool ok() const { return ok_; }
  T *get() const { return d_; }
  void download(T *dst, cudaStream_t stream) const {
    if (n_ != 0) {
      cudaMemcpyAsync(dst, d_, n_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }
  }

 private:
  T *d_ = nullptr;
  size_t n_ = 0;
  bool ok_ = false;
};

#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
/**
 * Dispatch the batched entry points by CTE wire id.
 *
 * The nine algorithms take identical parameters and differ only in the opts
 * struct, so this is a switch and nothing more. Keeping it in one place is
 * what makes the path codec-agnostic instead of LZ4-only.
 */
#define CLIO_NVC_CASE_TEMP(WID, ALGO)                                         \
  case WID:                                                                   \
    return nvcompBatched##ALGO##DecompressGetTempSizeSync(                    \
               cptr, csz, nch, max_unc, temp_bytes, 0,                        \
               nvcompBatched##ALGO##DecompressDefaultOpts, status, stream) == \
           nvcompSuccess;

bool NvcompBatchedTempSizeImpl(int wire, const void *const *cptr,
                               const size_t *csz, size_t nch, size_t max_unc,
                               size_t *temp_bytes, nvcompStatus_t *status,
                               cudaStream_t stream) {
  switch (wire) {
    CLIO_NVC_CASE_TEMP(11, LZ4)
    CLIO_NVC_CASE_TEMP(12, Snappy)
    CLIO_NVC_CASE_TEMP(13, Deflate)
    CLIO_NVC_CASE_TEMP(14, Gdeflate)
    CLIO_NVC_CASE_TEMP(15, Zstd)
    CLIO_NVC_CASE_TEMP(16, ANS)
    default:
      return false;
  }
}
#undef CLIO_NVC_CASE_TEMP

#define CLIO_NVC_CASE_DEC(WID, ALGO)                                          \
  case WID:                                                                   \
    return nvcompBatched##ALGO##DecompressAsync(                              \
               cptr, csz, obytes, osz, nch, temp, temp_bytes, optr,           \
               nvcompBatched##ALGO##DecompressDefaultOpts, status, stream) == \
           nvcompSuccess;

bool NvcompBatchedDecompressImpl(int wire, const void *const *cptr,
                                 const size_t *csz, const size_t *obytes,
                                 size_t *osz, size_t nch, void *temp,
                                 size_t temp_bytes, void *const *optr,
                                 nvcompStatus_t *status, cudaStream_t stream) {
  switch (wire) {
    CLIO_NVC_CASE_DEC(11, LZ4)
    CLIO_NVC_CASE_DEC(12, Snappy)
    CLIO_NVC_CASE_DEC(13, Deflate)
    CLIO_NVC_CASE_DEC(14, Gdeflate)
    CLIO_NVC_CASE_DEC(15, Zstd)
    CLIO_NVC_CASE_DEC(16, ANS)
    default:
      return false;
  }
}
#undef CLIO_NVC_CASE_DEC
#endif

/**
 * Chunk table of one stored blob: [CTEC 32B][nvcomp HLIF stream].
 *
 * The batched nvcomp entry points take flat arrays of chunk pointers and
 * sizes, not manager streams, so the HLIF chunk table has to be read out
 * first. Layout (fixed offsets, matches gpu_vector's mapper byte for byte):
 *
 *   CTEC 32B: +0 magic 'CTEC'  +4 codec  +16 orig(u64)
 *   HLIF @32: +0 magic(lo32)   +16 orig  +24 nchunks  +48 chunk_raw
 *             +56 dstart       +72 rel offsets[n]     +72+8n sizes[n]
 *   chunk c lives at blob + 32 + dstart + rel[c], for size[c] bytes.
 */
struct BlobChunks {
  unsigned long long orig = 0;       // decoded bytes
  unsigned long long chunk_raw = 0;  // uncompressed bytes per chunk
  std::vector<unsigned long long> rel;   // chunk offset from blob start
  std::vector<unsigned long long> csz;   // chunk compressed bytes
};

constexpr unsigned int kCtecMagic = 0x43544543u;
constexpr unsigned long long kHlifMagic = 1385239334ull;

/** Parse `hdr` (a host copy of the blob's first bytes). */
bool ParseBlobChunks(const unsigned char *hdr, size_t hdr_len,
                     size_t stored_size, BlobChunks *out) {
  if (hdr_len < 32 + 72) return false;
  unsigned int magic = 0;
  std::memcpy(&magic, hdr, 4);
  if (magic != kCtecMagic) return false;          // stored RAW, not encoded
  std::memcpy(&out->orig, hdr + 16, 8);
  if (out->orig == 0) return false;
  const unsigned char *hl = hdr + 32;
  unsigned long long v = 0;
  std::memcpy(&v, hl, 8);
  if ((v & 0xffffffffull) != kHlifMagic) return false;
  std::memcpy(&v, hl + 16, 8);
  if (v != out->orig) return false;
  unsigned long long nchunks = 0;
  std::memcpy(&nchunks, hl + 24, 8);
  if (nchunks == 0 || nchunks > (1ull << 20)) return false;
  std::memcpy(&out->chunk_raw, hl + 48, 8);
  if (out->chunk_raw == 0) return false;
  unsigned long long dstart = 0;
  std::memcpy(&dstart, hl + 56, 8);
  if (dstart != 72 + 16 * nchunks) return false;
  if (hdr_len < 32 + dstart) return false;        // table not fully copied
  out->rel.resize(nchunks);
  out->csz.resize(nchunks);
  for (unsigned long long c = 0; c < nchunks; ++c) {
    unsigned long long rel = 0, cs = 0;
    std::memcpy(&rel, hl + 72 + 8 * c, 8);
    std::memcpy(&cs, hl + 72 + 8 * nchunks + 8 * c, 8);
    if (cs == 0 || 32 + dstart + rel + cs > stored_size) return false;
    out->rel[c] = 32 + dstart + rel;              // from the blob's start
    out->csz[c] = cs;
  }
  return true;
}

}  // namespace

/**
 * Decompress MANY pages in ONE nvcomp launch.
 *
 * Every item is (stored blob in device memory) -> (page in device memory).
 * The chunk tables of all of them are flattened into a single set of arrays
 * and handed to nvcompBatched<Algo>DecompressAsync once, on the module's
 * stream, followed by ONE synchronize.
 *
 * Why this shape:
 *   - the CPU launches the codec; no device-side codec code, so nothing
 *     depends on nvcomp's device API being present or on LZ4 internals that
 *     other GPUs may not have (the batched entry points exist for all nine
 *     algorithms, with identical signatures, in the ordinary host library)
 *   - one launch for N pages instead of N launches each awaited. The manager
 *     API cannot do this: create_manager parses the header on the host and
 *     synchronizes, and Decompress synchronizes again, so it serializes at
 *     two points per page however it is called.
 *   - device pointer in, device pointer out. The only host traffic is the
 *     blob HEADERS, copied D2H to read their chunk tables.
 *
 * @return number of items decoded; `ok[i]` says which.
 */
void *Runtime::ModuleStream() {
#if CTP_ENABLE_GPU
  if (module_stream_ == nullptr) {
    module_stream_ = ctp::GpuApi::CreateStream();
  }
#endif
  return module_stream_;
}

size_t Runtime::GpuDecompressBatch(const std::vector<DecompItem> &items,
                                   std::vector<char> *ok) {
  ok->assign(items.size(), 0);
  if (items.empty()) return 0;
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
  CodecCtxGuard guard(codec_ctx_);
  if (!guard.ok()) return 0;
  cudaStream_t stream = static_cast<cudaStream_t>(ModuleStream());
  if (stream == nullptr) return 0;

  // 1. Read every blob's header and chunk table. This is the only host-side
  //    traffic: ~a few hundred bytes per page, never the payload.
  const size_t kHdrMax = 32 + 72 + 16 * 1024;   // enough for 1024 chunks
  std::vector<unsigned char> hdr(kHdrMax);
  std::vector<BlobChunks> parsed(items.size());
  std::vector<char> usable(items.size(), 0);
  int wire = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    const size_t want = std::min(items[i].stored_size, kHdrMax);
    if (cudaMemcpyAsync(hdr.data(), items[i].src_device, want,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      continue;
    }
    int w = 0;
    std::memcpy(&w, hdr.data() + 4, 4);
    if (wire == 0) wire = w;
    if (w != wire) continue;                    // mixed codecs: next batch
    if (!ParseBlobChunks(hdr.data(), want, items[i].stored_size, &parsed[i])) {
      continue;                                  // raw or unparseable
    }
    usable[i] = 1;
  }

  // 2. Flatten every page's chunks into one batch.
  std::vector<const void *> h_cptr;
  std::vector<size_t> h_csz, h_obytes;
  std::vector<void *> h_optr;
  std::vector<size_t> item_first(items.size(), 0), item_n(items.size(), 0);
  size_t max_unc = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    if (!usable[i]) continue;
    item_first[i] = h_cptr.size();
    const BlobChunks &b = parsed[i];
    for (size_t c = 0; c < b.rel.size(); ++c) {
      const unsigned long long done = c * b.chunk_raw;
      if (done >= b.orig) break;
      size_t want = static_cast<size_t>(b.orig - done);
      if (want > b.chunk_raw) want = static_cast<size_t>(b.chunk_raw);
      if (done + want > items[i].dst_bytes) break;
      h_cptr.push_back(static_cast<const char *>(items[i].src_device) +
                       b.rel[c]);
      h_csz.push_back(static_cast<size_t>(b.csz[c]));
      h_obytes.push_back(want);
      h_optr.push_back(static_cast<char *>(items[i].dst_device) + done);
      if (want > max_unc) max_unc = want;
      ++item_n[i];
    }
  }
  if (h_cptr.empty()) return 0;
  const size_t nch = h_cptr.size();

  // 3. Upload the descriptor arrays and run ONE batched decompress.
  DeviceArray<const void *> d_cptr(h_cptr, stream);
  DeviceArray<size_t> d_csz(h_csz, stream);
  DeviceArray<size_t> d_obytes(h_obytes, stream);
  DeviceArray<void *> d_optr(h_optr, stream);
  DeviceArray<size_t> d_osz(std::vector<size_t>(nch, 0), stream);
  DeviceArray<nvcompStatus_t> d_status(
      std::vector<nvcompStatus_t>(nch, nvcompSuccess), stream);
  if (!d_cptr.ok() || !d_csz.ok() || !d_obytes.ok() || !d_optr.ok() ||
      !d_osz.ok() || !d_status.ok()) {
    return 0;
  }
  size_t temp_bytes = 0;
  void *d_temp = nullptr;
  if (!NvcompBatchedTempSizeImpl(wire, d_cptr.get(), d_csz.get(), nch,
                                 max_unc, &temp_bytes, d_status.get(),
                                 stream)) {
    return 0;
  }
  if (temp_bytes != 0 && cudaMalloc(&d_temp, temp_bytes) != cudaSuccess) {
    return 0;
  }
  const bool launched = NvcompBatchedDecompressImpl(
      wire, d_cptr.get(), d_csz.get(), d_obytes.get(), d_osz.get(), nch,
      d_temp, temp_bytes, d_optr.get(), d_status.get(), stream);
  // ONE synchronize for the whole batch -- not one per page.
  const bool synced =
      launched && cudaStreamSynchronize(stream) == cudaSuccess;
  std::vector<size_t> osz(nch, 0);
  std::vector<nvcompStatus_t> st(nch, nvcompSuccess);
  if (synced) {
    d_osz.download(osz.data(), stream);
    d_status.download(st.data(), stream);
    cudaStreamSynchronize(stream);
  }
  if (d_temp != nullptr) cudaFree(d_temp);
  if (!synced) return 0;

  // 4. An item succeeded only if every one of its chunks did.
  size_t done = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    if (!usable[i] || item_n[i] == 0) continue;
    bool good = true;
    for (size_t c = 0; c < item_n[i]; ++c) {
      const size_t k = item_first[i] + c;
      if (st[k] != nvcompSuccess || osz[k] != h_obytes[k]) { good = false; break; }
    }
    if (good) { (*ok)[i] = 1; ++done; }
  }
  return done;
#else
  (void) items;
  return 0;
#endif
}

bool Runtime::GpuDecompressFromDevice(const char *stored_device,
                                      size_t stored_size, int wire_id,
                                      size_t payload, void *dst_device,
                                      size_t dst_bytes) {
  if (!HasCodecContext() || dst_bytes > codec_buf_bytes_) {
    return false;
  }
  const size_t hdr = sizeof(CompressionHeader);
  if (stored_size <= hdr || payload == 0 || !IsGpuCodec(wire_id)) {
    return false;
  }
  const size_t slot = AcquireCodecSlot();
  if (slot == SIZE_MAX) {
    return false;
  }
  CUstream sstream = static_cast<CUstream>(codec_slots_[slot].stream);
  size_t out = dst_bytes;
  bool ok = false;
  {
    CodecCtxGuard guard(codec_ctx_);
    if (guard.ok()) {
      ctp::CompressionFactory::SetGpuStreamForThread(sstream);
      auto codec = ctp::CompressionFactory::GetPreset(
          ctp::CompressionFactory::NameForWireId(wire_id),
          ctp::CompressionPreset::BALANCED);
      // Source and destination are both device memory: no staging buffer and
      // no copy, just the codec.
      ok = codec && codec->Decompress(dst_device, out,
                                      const_cast<char *>(stored_device) + hdr,
                                      payload);
      if (ok && cuStreamSynchronize(sstream) != CUDA_SUCCESS) {
        ok = false;
      }
      ctp::CompressionFactory::SetGpuStreamForThread(nullptr);
    }
  }
  ReleaseCodecSlot(slot);
  return ok;
}

bool Runtime::GpuDecompressToDevice(const char *stored_host, size_t stored_size,
                                    void *dst_device, size_t dst_bytes) {
  if (!HasCodecContext() || stored_size > codec_buf_bytes_ ||
      dst_bytes > codec_buf_bytes_) {
    return false;
  }
  const size_t hdr = sizeof(CompressionHeader);
  if (stored_size <= hdr) {
    return false;
  }
  const auto *header = reinterpret_cast<const CompressionHeader *>(stored_host);
  if (!header->IsValid() || !IsGpuCodec(header->compress_lib_)) {
    return false;
  }
  const size_t payload = (header->compressed_size_ != 0)
                             ? static_cast<size_t>(header->compressed_size_)
                             : (stored_size - hdr);
  const size_t slot = AcquireCodecSlot();
  if (slot == SIZE_MAX) {
    return false;  // all slots busy: caller uses the host path this time
  }
  void *sbuf = codec_slots_[slot].buf;
  CUstream sstream = static_cast<CUstream>(codec_slots_[slot].stream);
  size_t out = dst_bytes;
  bool ok = false;
  const auto t_beg = std::chrono::steady_clock::now();
  auto t_mid = t_beg;
  auto t_end = t_beg;
  {
    CodecCtxGuard guard(codec_ctx_);
    if (guard.ok()) {
      // Only the COMPRESSED bytes are staged; the page itself never moves.
      // ONE synchronize, not two. Every operation in this context waits for a
      // driver time slice (measured: a 44 KB H2D took 2.1 ms), so syncing
      // between the copy and the codec paid that wait twice per fault. The
      // copy is enqueued on the SAME stream the codec will use, so ordering is
      // guaranteed without a host-side wait in between.
      if (cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(sbuf),
                            stored_host + hdr, payload,
                            sstream) == CUDA_SUCCESS) {
        t_mid = std::chrono::steady_clock::now();
        // Bind the codec to THIS slot's stream; otherwise every concurrent
        // operation runs on the one stream the factory holds and they
        // serialize, which is what the per-slot buffers exist to prevent.
        ctp::CompressionFactory::SetGpuStreamForThread(sstream);
        auto codec = ctp::CompressionFactory::GetPreset(
            ctp::CompressionFactory::NameForWireId(header->compress_lib_),
            ctp::CompressionPreset::BALANCED);
        // DECOMPRESS DIRECTLY INTO THE FAULT'S PAGE. dst_device is the page the
        // faulting block is waiting on, exactly dst_bytes long, and managed, so
        // this context can write it. nvcomp writes into the caller's output
        // when it is device memory big enough -- no intermediate, no copy back.
        ok = codec && codec->Decompress(dst_device, out, sbuf, payload);
        ctp::CompressionFactory::SetGpuStreamForThread(nullptr);
        t_end = std::chrono::steady_clock::now();
      }
    }
  }
  ReleaseCodecSlot(slot);
  if (getenv("CLIO_CODEC_TIME")) {
    // Split the operation: how much is the H2D of the compressed bytes, how
    // much is the codec itself (which is where a context switch would land).
    static std::atomic<long long> n{0}, tot_us{0}, cop_us{0};
    const long long a_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_mid - t_beg).count();
    const long long b_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_end - t_mid).count();
    tot_us += a_us; cop_us += b_us;
    const long long c = ++n;
    if (c % 64 == 0) {
      fprintf(stderr, "[TIME] decomp n=%lld avg_h2d=%lldus avg_codec=%lldus\n",
              c, tot_us.load() / c, cop_us.load() / c);
      fflush(stderr);
    }
  }
  return ok;
}

/**
 * Compress device -> device: the compressed bytes land in `out_device` and
 * never touch host memory.
 *
 * Same codec call as GpuCompressFromDevice; the only difference is that the
 * result stays where PutBlob can write it straight to the device tier,
 * instead of being copied D2H only to be copied H2D again.
 */
bool Runtime::GpuCompressToDevice(int wire_id, const void *src_device,
                                  size_t size, char *out_device,
                                  size_t out_cap, size_t *out_size) {
  if (!HasCodecContext() || size > codec_buf_bytes_) {
    return false;
  }
  const size_t slot = AcquireCodecSlot();
  if (slot == SIZE_MAX) {
    return false;
  }
  void *sbuf = codec_slots_[slot].buf;
  CUstream sstream = static_cast<CUstream>(codec_slots_[slot].stream);
  size_t csize = codec_buf_bytes_;
  bool ok = false;
  {
    CodecCtxGuard guard(codec_ctx_);
    if (guard.ok()) {
      ctp::CompressionFactory::SetGpuStreamForThread(sstream);
      auto codec = ctp::CompressionFactory::GetPreset(
          ctp::CompressionFactory::NameForWireId(wire_id),
          ctp::CompressionPreset::BALANCED);
      if (codec && codec->Compress(sbuf, csize, const_cast<void *>(src_device),
                                   size) &&
          csize <= out_cap &&
          cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(out_device),
                            reinterpret_cast<CUdeviceptr>(sbuf), csize,
                            sstream) == CUDA_SUCCESS &&
          cuStreamSynchronize(sstream) == CUDA_SUCCESS) {
        ok = true;
      }
      ctp::CompressionFactory::SetGpuStreamForThread(nullptr);
    }
  }
  ReleaseCodecSlot(slot);
  if (ok) {
    *out_size = csize;
  }
  return ok;
}

bool Runtime::GpuCompressFromDevice(int wire_id, const void *src_device,
                                    size_t size, char *out_host, size_t out_cap,
                                    size_t *out_size) {
#define CTRACE(...) do { if (getenv("CLIO_CODEC_TRACE")) { \
    fprintf(stderr, "[TRACE] " __VA_ARGS__); fflush(stderr);} } while (0)
  if (!HasCodecContext() || size > codec_buf_bytes_) {
    CTRACE("comp: no ctx or too big (size=%zu cap=%zu)\n", size, codec_buf_bytes_);
    return false;
  }
  const size_t slot = AcquireCodecSlot();
  if (slot == SIZE_MAX) {
    CTRACE("comp: all slots busy\n");
    return false;
  }
  void *sbuf = codec_slots_[slot].buf;
  CUstream sstream = static_cast<CUstream>(codec_slots_[slot].stream);
  size_t csize = codec_buf_bytes_;
  bool ok = false;
  {
    CodecCtxGuard guard(codec_ctx_);
    if (guard.ok()) {
      ctp::CompressionFactory::SetGpuStreamForThread(sstream);
      auto codec = ctp::CompressionFactory::GetPreset(
          ctp::CompressionFactory::NameForWireId(wire_id),
          ctp::CompressionPreset::BALANCED);
      // Compress FROM the caller's page (managed, so readable here) INTO this
      // slot's device buffer. The page never leaves the device.
      if (codec && codec->Compress(sbuf, csize, const_cast<void *>(src_device),
                                   size) &&
          csize <= out_cap &&
          cuMemcpyDtoHAsync(out_host, reinterpret_cast<CUdeviceptr>(sbuf),
                            csize, sstream) == CUDA_SUCCESS &&
          cuStreamSynchronize(sstream) == CUDA_SUCCESS) {
        ok = true;
      }
      ctp::CompressionFactory::SetGpuStreamForThread(nullptr);
    }
  }
  ReleaseCodecSlot(slot);
  if (ok) {
    *out_size = csize;
    CTRACE("comp: OK csize=%zu\n", csize);
  } else {
    CTRACE("comp: failed\n");
  }
  return ok;
#undef CTRACE
}

#endif  // CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP

#if CTP_ENABLE_GPU
/**
 * ShmPtr addressing a RAW device pointer.
 *
 * A null allocator id means "off_ is already an absolute address", which is
 * how gpu_vector hands its device pages to PodPutBlob (DeviceVector::RawPtr).
 * Resolving it through an allocator instead yields a pointer the bdev's
 * cudaMemcpyAsync cannot read -- it segfaults inside the driver.
 */
ctp::ipc::ShmPtr<void> Runtime::ScratchShmPtr(char *p) const {
  ctp::ipc::ShmPtr<void> sp;
  sp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  sp.off_ = reinterpret_cast<clio::run::u64>(p);
  return sp;
}
#endif

char *Runtime::AcquireGpuScratch(size_t bytes) {
  std::lock_guard<std::mutex> guard(gpu_scratch_mu_);
  if (bytes > gpu_scratch_slab_ || gpu_scratch_free_.empty()) {
    return nullptr;
  }
  char *p = gpu_scratch_free_.back();
  gpu_scratch_free_.pop_back();
  return p;
}

void Runtime::ReleaseGpuScratch(char *ptr) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> guard(gpu_scratch_mu_);
  gpu_scratch_free_.push_back(ptr);
}

bool Runtime::IsGpuCodec(int wire_id) {
  if (wire_id <= 0) {
    return false;
  }
  const std::string name = ctp::CompressionFactory::NameForWireId(wire_id);
  return name.rfind("nvcomp-", 0) == 0;
}

int Runtime::CpuEquivalentCodec(int gpu_wire_id) {
  const std::string name =
      ctp::CompressionFactory::NameForWireId(gpu_wire_id);
  if (name.rfind("nvcomp-", 0) != 0) {
    return gpu_wire_id;
  }
  // "nvcomp-lz4" -> "lz4", and so on. gdeflate/deflate have no exact CPU twin
  // in the registry; zlib is the same DEFLATE bitstream family, and ans has
  // none at all, which GetWireId reports as 0 (store raw).
  std::string cpu = name.substr(7);
  if (cpu == "gdeflate" || cpu == "deflate") {
    cpu = "zlib";
  }
  const int id = ctp::CompressionFactory::GetWireId(cpu);
  return id;
}

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
  // Compress STRAIGHT INTO the shm buffer the bytes are headed for. Going via
  // a std::vector cost a full-payload heap allocation and a second pass over
  // the compressed bytes on every single blob -- per PAGE on the gpu_vector
  // path, where blobs are pages. The buffer is sized to the codec's worst-case
  // bound and only `total` bytes are ever handed to PutBlob, so the slack
  // costs an allocation size, not I/O.
  size_t header_size = sizeof(CompressionHeader);
  const size_t bound = size + (size / 20) + 1024;
  auto shm = CLIO_IPC->AllocateBuffer(header_size + bound);
  if (shm.IsNull()) {
    return false;
  }
  size_t compressed_size = bound;
  if (!compressor->Compress(shm.ptr_ + header_size, compressed_size,
                            const_cast<char *>(src), size)) {
    CLIO_IPC->FreeBuffer(shm);
    return false;
  }
  size_t total = compressed_size + header_size;
  // MEANINGFUL gain required, not any gain. Every reader of a compressed
  // blob pays a decompress; on the gpu_vector device-tier path that is an
  // in-kernel decode per page fault. Storing at 93% of raw was measured to
  // double a paged eval for a ~7% capacity win — a straight loss for any
  // consumer. An eighth is a deliberately conservative bar: genuinely
  // compressible data (<=87.5%) sails through, marginal data stays raw and
  // (on device tiers) zero-copy mappable.
  if (total >= size - size / 8) {
    CLIO_IPC->FreeBuffer(shm);
    return false;  // not beneficial enough — caller stores raw
  }
  CompressionHeader header(ctx.compress_lib_, ctx.compress_preset_, size,
                           compressed_size);
  std::memcpy(shm.ptr_, &header, header_size);
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

/**
 * POD PutBlob from a device producer (gpu_vector page flush).
 *
 * Same contract as PutBlob: whole-blob writes with a codec requested get
 * compressed, everything else forwards untouched. The difference is that
 * blob_data_ points at DEVICE memory, so the payload is staged to host before
 * the codec sees it -- a codec cannot read a device pointer.
 *
 * Without this handler the interposer's default case forwarded these straight
 * through, so pages written from a kernel were stored UNCOMPRESSED while
 * every layer reported success.
 */
clio::run::TaskResume Runtime::CompressPodPutBlob(
    clio::run::shared_ptr<clio::cte::core::PodPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::cte::core::Context &ctx = task->context_;
    if (ctx.compress_lib_ <= 0 || ctx.replica_ != 0 || task->offset_ != 0 ||
        task->size_ == 0) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodPutBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    auto src_full =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    if (src_full.ptr_ == nullptr) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodPutBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    // A GPU codec CANNOT run here. This handler services a gpu_vector page
    // fault, which means a kernel is resident on the device and spinning until
    // this returns; a GPU codec needs that same device to do its work, so it
    // blocks forever waiting for a kernel that is waiting for it. Measured
    // directly: NvComp::Compress was entered and never returned, and the
    // runtime's own stall detector reported the worker stuck on one task for
    // 87 seconds. Preallocating the stream and every buffer -- so nvcomp
    // allocates nothing at all -- does not help, because the cycle is over the
    // DEVICE, not over any allocator.
    //
    // Substitute the CPU codec of the same family rather than failing: the
    // header records the codec actually used, so readers stay correct, and the
    // caller gets compression instead of a hang. Making a GPU codec genuinely
    // work for a paging consumer needs decompression that does not contend
    // with the consumer kernel -- either driven off the fault path entirely,
    // or performed device-side inside the consumer itself.
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
    if (IsGpuCodec(ctx.compress_lib_) && HasCodecContext() &&
        ctp::IsDeviceAccessible(src_full.ptr_)) {
      const size_t sz = static_cast<size_t>(task->size_);
      const size_t bound = sz + (sz / 20) + 1024;
      const size_t hdr_sz = sizeof(CompressionHeader);
      // Compress DEVICE -> DEVICE and put from device memory. The compressed
      // bytes are destined for the kHBM tier, so staging them through a host
      // SHM buffer (as this did) meant a D2H copy followed immediately by an
      // H2D copy of the same bytes, on every page written.
      // WAIT for a scratch slab and a codec slot rather than giving up on
      // them. Both pools are small and every concurrent page put draws from
      // them, so transient exhaustion is normal -- it is not evidence that the
      // GPU cannot do the work, and treating it as such is how a CPU fallback
      // creeps back in. (Measured: 2 of 256 pages lost the race.)
      char *scratch = nullptr;
      size_t csize = 0;
      bool _gc = false;
      for (int attempt = 0; attempt < 10000; ++attempt) {
        if (scratch == nullptr) {
          scratch = AcquireGpuScratch(hdr_sz + bound);
        }
        if (scratch != nullptr) {
          _gc = GpuCompressToDevice(ctx.compress_lib_, src_full.ptr_, sz,
                                    scratch + hdr_sz, bound, &csize);
          if (_gc) break;
          // Compression itself can only fail here for want of a codec slot;
          // a genuine codec error would fail identically on retry and is
          // caught by the bounded loop.
        }
        CLIO_CO_AWAIT(clio::run::yield(50.0));
      }
      if (getenv("CLIO_CODEC_TRACE")) fprintf(stderr, "[TRACE] gpu_comp ok=%d csize=%zu sz=%zu\n", (int)_gc, csize, sz);
      if (_gc && csize + hdr_sz < sz) {
        // The header goes to the device too -- it is the first bytes of the
        // stored blob, and BuildDeviceTierMap parses it out of the tier.
        CompressionHeader header(ctx.compress_lib_, ctx.compress_preset_,
                                 task->size_, csize);
        // 32 bytes of METADATA, not payload: the compressed bytes themselves
        // are already on the device and stay there.
        ctp::DeviceAwareMemcpy(scratch, &header, hdr_sz);
        ctx.actual_original_size_ = task->size_;
        ctx.actual_compressed_size_ = csize + hdr_sz;
        ctx.actual_compression_ratio_ =
            static_cast<double>(sz) / static_cast<double>(csize + hdr_sz);
        ctx.transform_flags_ |= clio::cte::core::kBlobTransformed |
                                clio::cte::core::kBlobTransformCompressed;
        if (!core_client_) {
          core_client_ =
              std::make_unique<clio::cte::core::Client>(CorePoolId());
        }
        auto put = core_client_->AsyncPutBlob(
            task->tag_id_, task->blob_name_.str(), 0, csize + hdr_sz,
            ScratchShmPtr(scratch), task->score_, ctx, task->flags_,
            clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        task->context_ = put->context_;
        task->return_code_ = put->GetReturnCode();
        ReleaseGpuScratch(scratch);
        CLIO_CO_RETURN;
      }
      ReleaseGpuScratch(scratch);
    }
#endif
    if (IsGpuCodec(ctx.compress_lib_)) {
      // NO CPU SUBSTITUTION. This used to silently swap in the CPU codec of
      // the same family, on the theory that "a GPU codec cannot service a
      // device page fault (the faulting kernel holds the device)". Yieldable
      // kernels removed that premise: a faulting block suspends and the
      // kernel exits, so the device is free while the codec runs. Falling
      // back would produce a CPU-lz4 bitstream that the in-kernel decoder
      // cannot read, quietly forcing every later read onto the host path --
      // i.e. it would report a GPU codec while delivering a CPU one.
      HLOG(kError,
           "compressor: GPU codec '{}' requested but the device codec path "
           "did not run (codec context: {}, device src: {}). Failing the put "
           "rather than silently substituting a CPU codec.",
           ctp::CompressionFactory::NameForWireId(ctx.compress_lib_),
           HasCodecContext(), ctp::IsDeviceAccessible(src_full.ptr_));
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Stage device -> host. DeviceAwareMemcpy resolves the pointer kind, so
    // this is also correct when the producer handed us host memory.
    //
    // The buffer is per-thread and reused. A fresh std::vector per call meant
    // a page-sized heap allocation (and its first-touch page faults) on every
    // blob -- on the gpu_vector path that is once per PAGE, in the middle of a
    // synchronous device fault. Reuse is safe here because the buffer is
    // filled and consumed with no suspension in between: a fiber cannot yield
    // between this copy and CompressIntoShm below, so no other fiber on this
    // thread can observe it mid-use.
    static thread_local std::vector<char> host;
    if (host.size() < static_cast<size_t>(task->size_)) {
      host.resize(static_cast<size_t>(task->size_));
    }
    ctp::DeviceAwareMemcpy(host.data(), src_full.ptr_,
                           static_cast<size_t>(task->size_));

    ctp::ipc::FullPtr<char> stored;
    clio::run::u64 stored_size = 0;
    if (!CompressIntoShm(ctx, host.data(), task->size_, &stored,
                         &stored_size)) {
      // Not worth compressing (or the codec failed): store the raw bytes and
      // record NO codec, so a reader never tries to decompress them.
      ctx.compress_lib_ = 0;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodPutBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    if (!core_client_) {
      core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
    }
    auto put = core_client_->AsyncPutBlob(
        task->tag_id_, task->blob_name_.str(), 0, stored_size,
        stored.shm_.template Cast<void>(), task->score_, ctx, task->flags_,
        clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(put);
    task->context_ = put->context_;
    task->return_code_ = put->GetReturnCode();
    CLIO_IPC->FreeBuffer(stored);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * POD GetBlob for a device consumer (gpu_vector page fault).
 *
 * Reads the STORED bytes, decompresses when they carry our header, and lands
 * the result in the caller's DEVICE buffer. The stored form is detected from
 * the header magic rather than from the request, so a blob written raw
 * (because compression was not worth it) reads back correctly through the
 * same path.
 *
 * A request with NO codec is forwarded instead of interposed. That is not an
 * optimization of a rare case: it is the whole uncompressed path. Interposing
 * means allocating a host SHM buffer, reading the blob into it, and copying
 * that to the caller's device pointer -- storage -> host -> device. Forwarding
 * hands the caller's own destination pointer to the core, so the bdev copies
 * straight into it (MemBdevTransport::ReadBlocks dispatches on
 * IsDevicePointer), which is one transfer instead of two and is a
 * device-to-device copy whenever the tier holding the blob is device memory.
 * Compare CompressPodPutBlob, which has always forwarded the no-codec case;
 * the read side was interposing unconditionally, so simply composing the
 * compressor into a chain doubled the cost of every uncompressed device read.
 *
 * The codec is taken from the REQUEST, matching the put side: a reader asks
 * with the same context it wrote with. A blob stored raw because compression
 * did not pay still reads correctly when the request does carry a codec --
 * that case is detected below from the header magic, as before.
 */
clio::run::TaskResume Runtime::DecompressPodGetBlob(
    clio::run::shared_ptr<clio::cte::core::PodGetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    if (task->context_.compress_lib_ <= 0) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodGetBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    if (!core_client_) {
      core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
    }
    clio::run::u64 stored_size = 0;
    {
      auto sz = core_client_->AsyncGetBlobSize(task->tag_id_,
                                               task->blob_name_.str());
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
        task->return_code_ = sz->GetReturnCode() != 0 ? sz->GetReturnCode() : 1;
        CLIO_CO_RETURN;
      }
      stored_size = sz->size_;
    }
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
    // DEVICE-RESIDENT read of the stored bytes, when the destination is a
    // device page. Fetching them into a host buffer first means a D2H out of
    // the kHBM tier immediately followed by an H2D back into a codec slot --
    // the compressed payload crossing the bus twice to be decompressed on the
    // device it started on.
    {
      auto dst_probe =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      if (dst_probe.ptr_ != nullptr &&
          ctp::IsDeviceAccessible(dst_probe.ptr_)) {
        // FIRST: is the blob already in device memory? If it is on a kHbm
        // tier the bytes are right there, and fetching them into scratch
        // would copy device memory to device memory for nothing -- two task
        // round trips (GetBlobSize + GetBlob) to move data that never needed
        // to move. Decode straight from the tier instead.
        {
          unsigned long long lpool = 0, loff = 0, lsz = 0;
          if (clio_cte_locate(&task->tag_id_, task->blob_name_.str().c_str(),
                              &lpool, &loff, &lsz) == 0 &&
              lsz > sizeof(CompressionHeader)) {
            char *tier = clio_direct_dev_base(lpool);
            if (tier != nullptr) {
              CompressionHeader thdr;
              ctp::DeviceAwareMemcpy(&thdr, tier + loff, sizeof(thdr));
              if (thdr.IsValid() && IsGpuCodec(thdr.compress_lib_)) {
                auto req = std::make_shared<PendingDecomp>();
                req->src_device = tier + loff;
                req->stored_size = static_cast<size_t>(lsz);
                req->dst = dst_probe.ptr_;
                req->dst_bytes = static_cast<size_t>(task->size_);
                {
                  std::lock_guard<std::mutex> g(batch_mu_);
                  batch_.push_back(req);
                }
                int tspins = 0;
                while (!req->done.load(std::memory_order_acquire) &&
                       tspins < kDecompWaitMaxSpins) {
                  CLIO_CO_AWAIT(clio::run::yield(kDecompWaitPollUs));
                  ++tspins;
                }
                if (req->done.load(std::memory_order_acquire) && req->ok) {
                  task->return_code_ = 0;
                  CLIO_CO_RETURN;
                }
                req->abandoned.store(true, std::memory_order_release);
              }
            }
          }
        }
        char *dscratch = nullptr;
        for (int attempt = 0; attempt < 10000 && dscratch == nullptr;
             ++attempt) {
          dscratch = AcquireGpuScratch(stored_size);
          if (dscratch == nullptr) CLIO_CO_AWAIT(clio::run::yield(50.0));
        }
        if (dscratch != nullptr) {
          auto dget = core_client_->AsyncGetBlob(
              task->tag_id_, task->blob_name_.str(), 0, stored_size, 0,
              ScratchShmPtr(dscratch), clio::run::PoolQuery::Local());
          CLIO_CO_AWAIT(dget);
          if (dget->GetReturnCode() == 0) {
            // Only the 32-byte header comes back to the host, to be validated.
            CompressionHeader dhdr;
            ctp::DeviceAwareMemcpy(&dhdr, dscratch, sizeof(dhdr));
            if (dhdr.IsValid() && IsGpuCodec(dhdr.compress_lib_)) {
              // Hand it to the drainer instead of decompressing here. One
              // page decompressed on its own is one kernel launch and one
              // synchronize; the drainer collects whatever else is pending
              // and decodes all of it in a single launch.
              auto req = std::make_shared<PendingDecomp>();
              req->src_device = dscratch;
              req->stored_size = stored_size;
              req->dst = dst_probe.ptr_;
              req->dst_bytes = static_cast<size_t>(task->size_);
              {
                std::lock_guard<std::mutex> g(batch_mu_);
                batch_.push_back(req);
              }
              int spins = 0;
              while (!req->done.load(std::memory_order_acquire) &&
                     spins < kDecompWaitMaxSpins) {
                CLIO_CO_AWAIT(clio::run::yield(kDecompWaitPollUs));
                ++spins;
              }
              const bool got = req->done.load(std::memory_order_acquire) &&
                               req->ok;
              if (!got) req->abandoned.store(true, std::memory_order_release);
              ReleaseGpuScratch(dscratch);
              if (got) {
                task->return_code_ = 0;
                CLIO_CO_RETURN;
              }
              // Fall through to the paths below rather than returning a
              // half-decoded page.
            }
          }
          ReleaseGpuScratch(dscratch);
        }
      }
    }
#endif
    auto buf = CLIO_IPC->AllocateBuffer(stored_size);
    if (buf.IsNull()) {
      task->return_code_ = 4;
      CLIO_CO_RETURN;
    }
    {
      auto get = core_client_->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0, stored_size, 0,
          buf.shm_.template Cast<void>(), clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() != 0) {
        task->return_code_ = get->GetReturnCode();
        CLIO_IPC->FreeBuffer(buf);
        CLIO_CO_RETURN;
      }
    }
    auto dst_full =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    if (dst_full.ptr_ == nullptr) {
      task->return_code_ = 1;
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }

    const size_t hdr = sizeof(CompressionHeader);
    auto *header = reinterpret_cast<CompressionHeader *>(buf.ptr_);
#if CTP_ENABLE_GPU && CTP_ENABLE_NVCOMP
    // GPU codec: decompress in the codec context and peer-copy into the page.
    // Only the compressed bytes crossed the bus to get here; the page itself
    // never leaves the device.
    {
      auto dst_dev =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      const bool dev_ok =
          dst_dev.ptr_ != nullptr && ctp::IsDeviceAccessible(dst_dev.ptr_);
      if (dev_ok && batch_enabled_) {
        // Hand the page to the drainer and wait on our OWN flag only; nothing
        // here is responsible for another request's progress.
        auto req = std::make_shared<PendingDecomp>();
        // Own the bytes: the waiter may time out and free buf while the
        // drainer is still working (see PendingDecomp).
        req->stored_bytes.assign(buf.ptr_, buf.ptr_ + stored_size);
        req->stored_size = stored_size;
        req->dst = dst_dev.ptr_;
        req->dst_bytes = static_cast<size_t>(task->size_);
        {
          std::lock_guard<std::mutex> g(batch_mu_);
          batch_.push_back(req);
        }
        int spins = 0;
        while (!req->done.load(std::memory_order_acquire) &&
               spins < kDecompWaitMaxSpins) {
          CLIO_CO_AWAIT(clio::run::yield(kDecompWaitPollUs));
          ++spins;
        }
        if (!req->done.load(std::memory_order_acquire)) {
          req->abandoned.store(true, std::memory_order_release);
          if (getenv("CLIO_CODEC_TRACE")) {
            fprintf(stderr, "[DRAIN] waiter TIMEOUT\n");
            fflush(stderr);
          }
        } else if (req->ok) {
          CLIO_IPC->FreeBuffer(buf);
          task->return_code_ = 0;
          CLIO_CO_RETURN;
        }
      } else if (dev_ok &&
                 GpuDecompressToDevice(buf.ptr_, stored_size, dst_dev.ptr_,
                                       static_cast<size_t>(task->size_))) {
        CLIO_IPC->FreeBuffer(buf);
        task->return_code_ = 0;
        CLIO_CO_RETURN;
      }
    }
#endif
    if (stored_size > hdr && header->IsValid() &&
        IsGpuCodec(header->compress_lib_)) {
      // A GPU codec must NEVER run here. This is the faulting context, whose
      // kernel is spinning until this returns, so a codec that needs the device
      // waits on a kernel that is waiting on it. Reaching this point means the
      // batched path could not serve the page, and there is no CPU fallback --
      // an nvcomp bitstream is not a CPU-lz4 bitstream. Fail the read loudly
      // rather than deadlock the runtime.
      HLOG(kError,
           "compressor: GPU-compressed blob '{}' could not be served by the "
           "codec context; failing the read rather than deadlocking",
           task->blob_name_.str());
      task->return_code_ = 6;
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }
    if (stored_size > hdr && header->IsValid()) {
      std::string library_name =
          ctp::CompressionFactory::NameForWireId(header->compress_lib_);
      ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
      if (header->compress_preset_ == 1) {
        preset = ctp::CompressionPreset::FAST;
      } else if (header->compress_preset_ == 3) {
        preset = ctp::CompressionPreset::BEST;
      }
      auto codec = ctp::CompressionFactory::GetPreset(library_name, preset);
      if (!codec) {
        task->return_code_ = 3;
        CLIO_IPC->FreeBuffer(buf);
        CLIO_CO_RETURN;
      }
      // Exact payload length -- see CompressionHeader::compressed_size_.
      size_t payload = (header->compressed_size_ != 0)
                           ? static_cast<size_t>(header->compressed_size_)
                           : (stored_size - hdr);
      // Per-thread and reused, for the same reason as the put side: this runs
      // inside a synchronous device page fault, and a page-sized allocation
      // per fault is pure latency. Nothing suspends between the decompress
      // and the copy out, so the buffer cannot be observed mid-use.
      static thread_local std::vector<char> plain;
      const size_t original = static_cast<size_t>(header->original_size_);
      if (plain.size() < original) {
        plain.resize(original);
      }
      size_t out = original;
      if (!codec->Decompress(plain.data(), out, buf.ptr_ + hdr, payload) ||
          out != original) {
        task->return_code_ = 5;
        CLIO_IPC->FreeBuffer(buf);
        CLIO_CO_RETURN;
      }
      // `original`, NOT plain.size(): the buffer is reused and only grows, so
      // its size is a high-water mark from some earlier, larger blob. Copying
      // that many bytes would hand the caller the tail of a previous page.
      const size_t n = std::min<size_t>(task->size_, original);
      ctp::DeviceAwareMemcpy(dst_full.ptr_, plain.data(), n);
    } else {
      // Stored raw: hand the bytes back untouched.
      const size_t n = std::min<size_t>(task->size_, stored_size);
      ctp::DeviceAwareMemcpy(dst_full.ptr_, buf.ptr_, n);
    }
    CLIO_IPC->FreeBuffer(buf);
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}


// Batched POD paging. Fan each record into the scalar Pod task and run it
// through the scalar handler, so every page gets byte-identical compression
// treatment (codec choice, header, GPU-codec substitution) and there is exactly
// one place where that logic lives. The batch's payoff is on the submission
// side: the device enqueues one task for up to kPodMultiMax pages.
clio::run::TaskResume Runtime::CompressPodMultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::PodMultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    auto *ipc_manager = CLIO_CPU_IPC;
    task->num_ok_ = 0;
    int first_rc = 0;
    clio::run::u32 n = task->count_;
    if (n > clio::cte::core::kPodMultiMax) n = clio::cte::core::kPodMultiMax;
    for (clio::run::u32 i = 0; i < n; ++i) {
      auto &req = task->reqs_[i];
      auto sub = ipc_manager->NewTask<clio::cte::core::PodPutBlobTask>(
          clio::run::CreateTaskId(), task->pool_id_,
          clio::run::PoolQuery::Local(), task->tag_id_,
          req.blob_name_.c_str(), req.offset_, req.size_, req.data_,
          req.score_, task->context_, task->flags_);
      sub.get()->BeginRunContext();
      CLIO_CO_AWAIT(CompressPodPutBlob(sub));
      int rc = sub->GetReturnCode();
      req.rc_ = static_cast<clio::run::u32>(rc);
      if (rc == 0) {
        task->num_ok_++;
      } else if (first_rc == 0) {
        first_rc = rc;
      }
    }
    task->SetReturnCode(first_rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DecompressPodMultiGetBlob(
    clio::run::shared_ptr<clio::cte::core::PodMultiGetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    auto *ipc_manager = CLIO_CPU_IPC;
    task->num_ok_ = 0;
    int first_rc = 0;
    clio::run::u32 n = task->count_;
    if (n > clio::cte::core::kPodMultiMax) n = clio::cte::core::kPodMultiMax;
    for (clio::run::u32 i = 0; i < n; ++i) {
      auto &req = task->reqs_[i];
      auto sub = ipc_manager->NewTask<clio::cte::core::PodGetBlobTask>(
          clio::run::CreateTaskId(), task->pool_id_,
          clio::run::PoolQuery::Local(), task->tag_id_,
          req.blob_name_.c_str(), req.offset_, req.size_, task->flags_,
          req.data_, task->context_);
      sub.get()->BeginRunContext();
      CLIO_CO_AWAIT(DecompressPodGetBlob(sub));
      int rc = sub->GetReturnCode();
      req.rc_ = static_cast<clio::run::u32>(rc);
      if (rc == 0) {
        task->num_ok_++;
      } else if (first_rc == 0) {
        first_rc = rc;
      }
    }
    task->SetReturnCode(first_rc);
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
